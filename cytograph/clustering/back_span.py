from typing import List
import numpy as np
from cytograph.utils import div0
import logging
import scipy.sparse as sparse
import shoji
from sklearn.utils.extmath import randomized_svd
from cytograph import requires, creates, Algorithm
from sklearn.utils.sparsefuncs import mean_variance_axis

# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36940.pdf


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


def insert_node(btree, n):
	if btree == []:
		return [n, n + 1]
	a, b = btree
	if isinstance(a, list):
		a = insert_node(a, n)
	else:
		if a > n:
			a += 1
		elif a == n:
			a = [n, n + 1]
	if isinstance(b, list):
		b = insert_node(b, n)
	else:
		if b > n:
			b += 1
		elif b == n:
			b = [n, n + 1]
	return [a, b]


class BackSPAN(Algorithm):
	def __init__(self, n_genes: int = 500, min_modularity: float = 0, batch_keys: List[str] = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.n_genes = n_genes
		self.min_modularity = min_modularity
		self.batch_keys = batch_keys
		
		self.data: sparse.csr_matrix = None
		self.labels: np.ndarray = None
		self.linkage = []
		self.stack = []

	def split(self, label_to_split):
		assert isinstance(self.data, sparse.csr_matrix), " BackSPAN: input matrix must be sparse.csr_matrix"
		data = self.data[self.labels == label_to_split]
		n_cells, _ = data.shape
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
		b = xcsc[:, genes]  # corresponds to B2 in the SPAN paper, but we use Pearson residuals instead of tf-idf

		# TODO: Harmonize here
		einv = 1 / sparse.linalg.norm(b, axis=1)
		if not np.all(np.isfinite(einv)):
			logging.warning(" BackSPAN: Not splitting because some cells are all-zero (to fix, increase n_genes or remove all-zero cells)")
			return

		b.data *= np.take(einv, b.indices)  # b.indices are the indices of the rows

		# Now calculate B(B.T * 1), but B.T * 1 is just the sum of columns; some fiddling with matrix types needed
		a = np.array(b @ b.T.sum(axis=1))[:, 0]

		temp = np.power(a, -0.5)

		d = sparse.diags(temp)
		c = d @ b

		u, s, vt = randomized_svd(c, n_components=2, n_iter=10, random_state=None)
		labels = (u.T[1] > 0).astype("uint32")  # Labels are the signs of 2nd left-singular vector

		Q = newman_girvan_modularity(b, labels)
		if Q <= self.min_modularity:
			logging.info(f" BackSPAN: Not splitting {n_cells} cells with Q = {Q:.2} <= {self.min_modularity:.2}")
		else:
			logging.info(f" BackSPAN: Splitting {n_cells} -> ({(labels == 0).sum()}, {(labels == 1).sum()}) cells with Q == {Q:.2} > {self.min_modularity:.2}")
			self.labels[self.labels > label_to_split] += 1
			self.labels[self.labels == label_to_split] = labels + label_to_split
			self.stack.append(label_to_split)
			self.stack.append(label_to_split + 1)
			self.linkage = insert_node(self.linkage, label_to_split)
			
	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("ValidGenes", "bool", ("genes",))
	@creates("Clusters", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		logging.info(" BackSPAN: Loading expression matrix")
		self.data = ws.Expression.sparse(cols=ws.ValidGenes[:]).tocsr()

		logging.info(" BackSPAN: Computing normalized cuts")
		self.labels = np.zeros(self.data.shape[0], dtype="uint32")
		self.stack.append(0)
		while len(self.stack) > 0:
			self.split(self.stack.pop())
		return self.labels
