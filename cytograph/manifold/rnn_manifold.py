import cytograph as cg
from cytograph import creates, requires, Algorithm
import logging
import numpy as np
import scipy.sparse as sparse
import shoji


class RnnManifold(Algorithm):
	"""
	Compute a radius nearest-neighbor manifold graph
	"""
	def __init__(self, k: int, metric: str, mutual: bool = False, max_radius_percentile: int = 90, **kwargs) -> None:
		"""
		Compute a radius nearest-neighbor manifold graph
	
		Args:
			k:                     The maximum number of neighbors to use
			metric:                The metric to use (e.g. "euclidean")
			mutual:                If true, use only mutual neighbors
			max_radius_percentile: The radius to use, expressed as a percentile
		"""
		super().__init__(**kwargs)
		self.k = k
		self.metric = metric
		self.mutual = mutual
		self.max_radius_percentile = max_radius_percentile

	@requires("Factors", "float32", ("cells", None))
	@creates("ManifoldRadius", "float32", ())
	@creates("ManifoldIndices", "uint32", (None, 2))
	@creates("ManifoldWeights", "float32", (None, ))
	def fit(self, ws: shoji.WorkspaceManager, save=False):
		logging.info(" RnnManifold: Loading data")
		data = self.Factors[:]
		logging.info(f" RnnManifold: Computing balanced KNN (k = {self.k}, metric = '{self.metric}')")
		bnn = cg.BalancedKNN(k=self.k, metric=self.metric, maxl=2 * self.k, sight_k=2 * self.k, n_jobs=-1)
		bnn.fit(data)
		logging.info(f" RnnManifold: Converting to graph")
		knn = bnn.kneighbors_graph(mode='distance')
		logging.info(f" RnnManifold: Removing self-edges and zeros")
		knn.setdiag(0)
		knn.eliminate_zeros()
		logging.info(f" RnnManifold: Converting to similarities")
		max_d = knn.data.max()
		knn.data = (max_d - knn.data) / max_d
		logging.info(f" RnnManifold: Computing resolution")
		radius = np.percentile(1 - knn.data, self.max_radius_percentile)
		logging.info(f" RnnManifold: {self.max_radius_percentile}th percentile radius: {radius:.02}")

		if self.mutual:
			logging.info(f" RnnManifold: Converting to mutual nearest neighbors")
			mknn = knn.minimum(knn.transpose())
			# Convert distances to similarities
			mknn.data = (max_d - mknn.data) / max_d
			mknn = mknn.tocoo()
			mknn.setdiag(0)
			inside = mknn.data > 1 - radius
			rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)
		else:
			knn = knn.tocoo()
			inside = knn.data > 1 - radius
			logging.info(f" RnnManifold: Creating sparse matrix")
			rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
		logging.info(f" RnnManifold: Total of {rnn.getnnz():,} edges")
		indices = np.vstack([rnn.row, rnn.col]).T
		logging.info(f" RnnManifold: Saving")
		return (radius, indices, rnn.data)
