from tabnanny import verbose
from typing import Tuple, List
import numpy as np
from cytograph.utils import div0
import logging
import scipy.sparse as sparse
import shoji
from sklearn.utils.extmath import randomized_svd
from cytograph import requires, creates, Algorithm
from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy.cluster.hierarchy import ClusterNode
from copy import deepcopy
import sys
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from harmony import harmonize


# Suppressing negative residuals
#   This makes it possible to use sparse residuals matrices (because zeros are always negative)
# Batch-aware Poisson residuals
# Using SPAN to compute optimal normalized cuts
#   This uses cosine distance which now makes sense because we use residuals
#   Exploits the fact that we now have sparse residuals
#   Efficiently uses the (implicit) full distance matrix
#   No need for KNN or a specific K
#   No need for gene selection (but see below)
# Selecting a new subspace at every branch of the SPAN algorithm
#   Including recomputing the resdiuals because gene saliency changes
#   Stopping when the cut is not different from a random cut
#   Prune the clustering tree to yield different resolutions
# Integration built-in based on Harmony (but on residuals)


def newman_girvan_modularity(b, labels):
	bsum = b.T.sum(axis=1)
	bsum0 = b[labels == 0].T.sum(axis=1)
	bsum1 = b[labels == 1].T.sum(axis=1)
	n = b.shape[0]

	L = bsum.T @ bsum - n
	O00 = bsum0.T @ bsum0 - n
	O11 = bsum1.T @ bsum1 - n
	L0 = bsum0.T @ bsum - n
	L1 = bsum1.T @ bsum - n

	Q = O00 / L - (L0 / L) ** 2 + O11 / L - (L1 / L) ** 2

	return Q.item()  # Q is a matrix of a scalar, so we must unbox it


class Stockholm(Algorithm):
	def __init__(self, n_genes: int = 500, min_modularity: float = 0, min_cells: int = 30, min_clusters: int = 2, batch_keys: List[str] = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.n_genes = n_genes
		self.min_modularity = min_modularity
		self.min_cells = min_cells
		if self.min_cells < 30 and batch_keys is not None:
			raise ValueError("min_cells must be equal or greater than 30 if batch_keys is not None (because Harmony does not work with clusters smaller than 30 cells)")
		self.min_clusters = min_clusters
		self.batch_keys = batch_keys
		
		self.data: sparse.csr_matrix = None
		self.labels: np.ndarray = None
		self.tree = None
		self.stack = []

	def _insert_node(self, tree: ClusterNode, n, Q, left_count, right_count):
		if tree.is_leaf():
			if tree.id > n:
				return ClusterNode(tree.id + 1, tree.left, tree.right, tree.dist, tree.count)
			elif tree.id == n:
				return ClusterNode(999_999, left=ClusterNode(n, count=left_count), right=ClusterNode(n + 1, count=right_count), dist=Q, count=left_count + right_count)
			else:
				return tree
		else:
			left = self._insert_node(tree.left, n, Q, left_count, right_count)
			right = self._insert_node(tree.right, n, Q, left_count, right_count)
			return ClusterNode(tree.id, left, right, tree.dist, tree.count)

	def _to_linkage(self, n_clusters):
		tree = deepcopy(self.tree)

		# Make the distances cumulative from the leaves
		def cumulate(tr):
			if not tr.is_leaf():
				tr.dist += max(cumulate(tr.left), cumulate(tr.right))
			return tr.dist
		cumulate(tree)

		linkage = []

		# Remove all nodes that merge two leaves, and place them on the linkage list
		def prune(tr):
			if tr.left.is_leaf() and tr.right.is_leaf():
				linkage.append((tr.left.id, tr.right.id, tr.dist, tr.count))
				tr.id = n_clusters + len(linkage) - 1
				tr.left = None
				tr.right = None
			if tr.left is not None and not tr.left.is_leaf():
				prune(tr.left)
			if tr.right is not None and not tr.right.is_leaf():
				prune(tr.right)

		while not tree.is_leaf():
			prune(tree)

		return np.array(linkage, dtype="float32")

	def _split(self, label_to_split, keys_df = None):
		assert isinstance(self.data, sparse.csr_matrix), " Stockholm: input matrix must be sparse.csr_matrix"
		n_cells = (self.labels == label_to_split).sum()
		if n_cells < self.min_cells:
			logging.info(f" Stockholm: Not splitting {n_cells} < {self.min_cells} cells ")
			return
		data = self.data[self.labels == label_to_split]
		x = data.tocoo()

		# Positive pearson residuals for non-zero elements only
		cell_totals = x.sum(axis=1).A1
		gene_totals = x.sum(axis=0).A1
		overall_totals = cell_totals.sum()
		expected = cell_totals[x.row] * gene_totals[x.col] / overall_totals
		residuals = div0((x.data - expected), np.sqrt(expected + np.power(expected, 2) / 100))
		residuals = np.clip(residuals, 0, np.sqrt(n_cells))
		xcsc = sparse.csc_matrix((residuals, (x.row, x.col)), dtype="float32")

		# Select genes by residuals variance
		var, _ = mean_variance_axis(xcsc, axis=0)
		genes = np.argsort(-var)[:self.n_genes]

		# Start the SPAN calculation
		# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36940.pdf
		if keys_df is not None:
			keys = keys_df[self.labels == label_to_split]
			if len(np.unique(keys)) <= 1:
				b = xcsc[:, genes]
			else:
				transformed = harmonize(xcsc[:, genes].toarray(), keys, batch_key=self.batch_keys, tol_harmony=1e-5, verbose=False)
				b = sparse.csc_matrix(np.clip(transformed, 0, np.sqrt(n_cells)))
		else:
			b = xcsc[:, genes]  # corresponds to B2 in the SPAN paper, but we use Pearson residuals instead of tf-idf

		einv = 1 / sparse.linalg.norm(b, axis=1)
		if not np.all(np.isfinite(einv)):
			logging.warning(" Stockholm: Not splitting because some cells are all-zero (to fix, increase n_genes or remove all-zero cells)")
			return

		b.data *= np.take(einv, b.indices)  # b.indices are the indices of the rows

		# Now calculate B(B.T * 1), but B.T * 1 is just the sum of columns
		a = np.array(b @ b.T.sum(axis=1))[:, 0]

		temp = np.power(a, -0.5)

		d = sparse.diags(temp)
		c = d @ b

		u, s, vt = randomized_svd(c, n_components=2, n_iter=10, random_state=None)
		labels = (u.T[1] > 0).astype("uint32")  # Labels are the signs of 2nd left-singular vector

		Q = newman_girvan_modularity(b, labels)
		n_clusters = self.labels.max() + 1
		if Q <= self.min_modularity and n_clusters >= self.min_clusters:
			logging.info(f" Stockholm: Not splitting {n_cells} cells with Q = {Q:.2} <= {self.min_modularity:.2}")
			return
		else:
			logging.info(f" Stockholm: Splitting {n_cells} -> ({(labels == 0).sum()}, {(labels == 1).sum()}) cells with Q == {Q:.2} > {self.min_modularity:.2}")
			self.labels[self.labels > label_to_split] += 1
			self.labels[self.labels == label_to_split] = labels + label_to_split
			self.stack.append(label_to_split)
			self.stack.append(label_to_split + 1)
			if self.tree is None:
				self.tree = ClusterNode(999_999, ClusterNode(0, count=1), ClusterNode(1, count=1), dist=Q, count=2)
			else:
				self.tree = self._insert_node(self.tree, label_to_split, Q, 1, 1)

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("ValidGenes", "bool", ("genes",))
	@creates("Clusters", "uint32", ("cells",))
	@creates("StockholmLinkage", "float32", (None, 4))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		logging.info(" Stockholm: Loading expression matrix")
		self.data = ws.Expression.sparse(cols=ws.ValidGenes[:]).tocsr()

		keys_df = None
		if self.batch_keys is not None and len(self.batch_keys) > 0:
			logging.info(f" Stockholm: Harmonizing based on batch keys {self.batch_keys}")
			keys_df = pd.DataFrame.from_dict({k: ws[k][:] for k in self.batch_keys})

		logging.info(" Stockholm: Computing normalized cuts")
		self.labels = np.zeros(self.data.shape[0], dtype="uint32")
		self.stack.append(0)
		while len(self.stack) > 0:
			self._split(self.stack.pop(), keys_df)
		return self.labels, self._to_linkage(self.labels.max() + 1)
