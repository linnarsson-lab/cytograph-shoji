from typing import Tuple, List
import numpy as np
from cytograph.utils import available_cpu_count
import logging
import scipy.sparse as sparse
import shoji
from sklearn.utils.extmath import randomized_svd
from cytograph import requires, creates, Algorithm
from ..species import Species
import tomotopy as tt
from scipy.cluster.hierarchy import ClusterNode, cut_tree
from copy import deepcopy
import pandas as pd
from harmony import harmonize


# Using LDA to reduce dimensionality without feature selection
# Using SPAN to compute optimal normalized cuts
#   This uses cosine distance
#   Efficiently uses the (implicit) full distance matrix
#   No need for KNN or a specific K
#   No need for gene selection (but see below)
# Selecting a new subspace at every branch of the SPAN algorithm
#   Including recomputing the LDA because gene saliency changes
#   Stopping when the cut is not different from a random cut
#   Prune the clustering tree to yield different resolutions
# Integration built-in based on Harmony (but on LDA topics)


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


def _recount(linkage):
	n_clusters = len(linkage) + 1
	for ix in range(len(linkage)):
		a, b, _, _ = linkage[ix, :]
		a = int(a)
		b = int(b)
		count = 0
		if a < n_clusters:
			count += 1
		else:
			count += linkage[a - n_clusters, 3]
		if b < n_clusters:
			count += 1
		else:
			count += linkage[b - n_clusters, 3]
		linkage[ix, 3] = count


def _pop_leaf(linkage):
	n_clusters = len(linkage) + 1
	nearest = np.argmin(linkage[:, 2])
	a, b, _, _ = linkage[nearest, :]
	linkage = np.delete(linkage, nearest, axis=0)
	
	linkage[:, :2][linkage[:, :2] == float(nearest + n_clusters)] = min(a, b)
	r1 = linkage[:, :2] > max(a, b)
	r2 = linkage[:, :2] > nearest + n_clusters
	linkage[:, :2][r1] -= 1
	linkage[:, :2][r2] -= 1

	_recount(linkage)
	return linkage


def _to_linkage(tree):
	tree = deepcopy(tree)
	n_clusters = tree.get_count()

	# Make the distances cumulative from the leaves, and assign IDs to the leaves
	i = 0

	def cumulate(tr):
		nonlocal i
		if tr.is_leaf():
			tr.id = i
			i += 1
		else:
			tr.dist += max(cumulate(tr.left), cumulate(tr.right))
		return tr.dist
	cumulate(tree)

	linkage = []

	# Remove all nodes that merge two leaves, and place them on the linkage list
	def merge(tr):
		if tr.left.is_leaf() and tr.right.is_leaf():
			linkage.append((tr.left.id, tr.right.id, tr.dist, tr.count))
			tr.id = n_clusters + len(linkage) - 1
			tr.left = None
			tr.right = None
		if tr.left is not None and not tr.left.is_leaf():
			merge(tr.left)
		if tr.right is not None and not tr.right.is_leaf():
			merge(tr.right)

	while not tree.is_leaf():
		merge(tree)

	return np.array(linkage, dtype="float32")


class StockholmLDA(Algorithm):
	def __init__(self, cut_at: float = 0.1, min_Q = 0.01, min_cells: int = 100, mask: List[str] = None, batch_keys: List[str] = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.cut_at = cut_at
		self.min_Q = min_Q
		self.min_cells = min_cells
		self.mask = mask if mask is not None else []
		self.masked_genes: np.ndarray = None

		if self.min_cells < 30 and batch_keys is not None:
			raise ValueError("min_cells must be equal or greater than 30 if batch_keys is not None (because Harmony does not work with clusters smaller than 30 cells)")
		self.batch_keys = batch_keys
		self.keys_df = None
		self.accessions = None
		self.data: sparse.csr_matrix = None
		self.labels: np.ndarray = None
		self.tree = None
		self.stack = []
		self.indent = 0

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

	def _split(self, label_to_split):
		assert isinstance(self.data, sparse.csr_matrix), " Stockholm: input matrix must be sparse.csr_matrix"
		n_cells = (self.labels == label_to_split).sum()
		if n_cells < self.min_cells:
			logging.info(f" Stockholm: {'│' * self.indent}└{n_cells} < {self.min_cells} cells ")
			self.indent -= 1
			return
		data = self.data[self.labels == label_to_split]

		# Latent Dirichlet allocation on full set of valid genes, using online variational bayes
		k = max(2, int(10 * (np.log10(n_cells) - 2)))
		logging.info(f" Stockholm: {'│' * (self.indent + 1)}LDA of {n_cells} cells with {k} topics")
		corpus = tt.utils.Corpus()
		for i in range(data.shape[0]):
			corpus.add_doc(list(np.repeat(self.accessions, data[i, :].A[0])))

		model = tt.LDAModel(k=k, alpha=50 / k, eta=0.1)
		model.add_corpus(corpus)
		last_perp = 0
		for i in np.arange(0, 500, 10):
			model.train(iter=10, workers=available_cpu_count(), parallel=tt.ParallelScheme.PARTITION)
			perp = model.perplexity
			# Escape loop when perplexity is stagnant
			if np.abs((last_perp - perp) / perp) < 0.01:
				break
			last_perp = perp
		y = np.vstack([doc.get_topic_dist() for doc in model.docs])

		# Harmonize
		if self.keys_df is not None:
			keys = self.keys_df[self.labels == label_to_split]
			if len(np.unique(keys)) > 1:
				# Harmonize in log space
				assert np.all(y > 0), "y must be strictly positive"
				logging.info(f" Stockholm: {'│' * (self.indent + 1)}Harmonizing")
				y = harmonize(y, keys, batch_key=self.batch_keys, tol_harmony=1e-5, verbose=False)
				max_ = np.max(y, axis=0)
				min_ = np.clip(np.min(y, axis=0), None, 0)  # Only rescale if topic extends to negative values
				y = np.clip((y - min_) * (max_ / (max_ - min_)), 0, None)  # Rescale to avoid negative values and clip for good measure

		# Start the SPAN calculation
		# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36940.pdf
		b = sparse.csc_matrix(y)
		einv = 1 / sparse.linalg.norm(b, axis=1)
		if not np.all(np.isfinite(einv)):
			logging.warning(f" Stockholm: {'│' * self.indent}└Not splitting because some cells are all-zero (to fix, increase n_genes or remove all-zero cells)")
			self.indent -= 1
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

		if Q <= self.min_Q:
			logging.info(f" Stockholm: {'│' * self.indent}├{n_cells} | Q = {Q:.2} <= {self.min_Q}")
			self.indent -= 1
			return
		else:
			logging.info(f" Stockholm: {'│' * self.indent}├{n_cells} -> ({(labels == 0).sum()}, {(labels == 1).sum()}) | Q == {Q:.2} > {self.min_Q}")
			self.indent += 1
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
	@requires("Accession", "string", ("genes",))
	@requires("Species", "string", ())
	@creates("Clusters", "uint32", ("cells",))
	@creates("StockholmLinkage", "float32", (None, 4))
	@creates("ClustersFine", "uint32", ("cells",))
	@creates("StockholmLinkageFine", "float32", (None, 4))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		species = Species(self.Species[:])
		self.masked_genes = species.mask(ws, self.mask)
		self.accessions = self.Accession[ws.ValidGenes == True]

		logging.info(" Stockholm: Loading expression matrix")
		self.data = ws.Expression.sparse(cols=ws.ValidGenes[:]).tocsr()

		if self.batch_keys is not None and len(self.batch_keys) > 0:
			logging.info(f" Stockholm: Will harmonize based on {self.batch_keys}")
			self.keys_df = pd.DataFrame.from_dict({k: ws[k][:] for k in self.batch_keys})

		logging.info(" Stockholm: Computing normalized cuts")
		self.labels = np.zeros(self.data.shape[0], dtype="uint32")
		self.stack.append(0)
		while len(self.stack) > 0:
			self._split(self.stack.pop())

		logging.info(f" Stockholm: Found {self.labels.max() + 1} preliminary clusters with min_Q={self.min_Q}")
		linkage = _to_linkage(self.tree)
		logging.info(f" Stockholm: Cutting dendrogram at {self.cut_at}")
		labels = cut_tree(linkage.astype("float64"), height=self.cut_at)[self.labels].flatten()  # need to flatten because it's a column vector
		logging.info(f" Stockholm: Found {labels.max() + 1} clusters")
		original_linkage = linkage
		while np.min(linkage[:, 2]) < self.cut_at:
			linkage = _pop_leaf(linkage)

		return labels, linkage, self.labels, original_linkage
