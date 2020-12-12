import cytograph as cg
from cytograph import creates, requires, Module
import logging
import numpy as np
import scipy.sparse as sparse
import shoji


class RnnManifold(Module):
	def __init__(self, k: int, metric: str, **kwargs) -> None:
		"""
		Args:
			k				The maximum number of neighbors to use
			metric			The metric to use (e.g. "euclidean")
		"""
		super().__init__(**kwargs)
		self.k = k
		self.metric = metric

	@requires("Factors", "float32", ("cells", None))
	@creates("ManifoldRadius", "float32", ())
	@creates("ManifoldIndices", "uint32", (None, 2))
	@creates("ManifoldWeights", "float32", (None, ))
	def fit(self, ws: shoji.WorkspaceManager, save=False):
		Factors = self.requires["Factors"]

		logging.info(" RnnManifold: Loading data")
		data = ws[Factors][...]
		logging.info(f" RnnManifold: Computing balanced KNN (k = {self.k}, metric = '{self.metric}')")
		bnn = cg.BalancedKNN(k=self.k, metric=self.metric, maxl=2 * self.k, sight_k=2 * self.k, n_jobs=-1)
		bnn.fit(data)
		logging.info(f" RnnManifold: Converting to graph")
		knn = bnn.kneighbors_graph(mode='distance')
		logging.info(f" RnnManifold: Removing zeros")
		knn.eliminate_zeros()
		logging.info(f" RnnManifold: Converting to mutual nearest neighbors")
		mknn = knn.minimum(knn.transpose())
		# Convert distances to similarities
		logging.info(f" RnnManifold: Converting to similarities")
		max_d = knn.data.max()
		knn.data = (max_d - knn.data) / max_d
		mknn.data = (max_d - mknn.data) / max_d
		mknn = mknn.tocoo()
		mknn.setdiag(0)
		# Compute the effective resolution
		logging.info(f" RnnManifold: Computing resolution")
		d = 1 - knn.data
		radius = np.percentile(d, 90)
		logging.info(f" RnnManifold: 90th percentile radius: {radius:.02}")
		inside = mknn.data > 1 - radius
		rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)

		return (radius, np.vstack([rnn.row, rnn.col]).T, rnn.data)
