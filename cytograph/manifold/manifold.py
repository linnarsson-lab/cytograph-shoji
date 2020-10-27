import cytograph as cg
from cytograph import creates
import logging
import numpy as np
import scipy.sparse as sparse
import shoji


class Manifold:
	def __init__(self, k: int, metric: str) -> None:
		self.k = k
		self.metric = metric

	@creates("RnnRadius", "float32", ())
	@creates("RNN_row", "uint32", (None, ))
	@creates("RNN_col", "uint32", (None, ))
	@creates("RNN_data", "float32", (None, ))
	def fit(self, ws: shoji.WorkspaceManager, data: np.ndarray, save=False):
		logging.info(f"Manifold: Computing balanced KNN (k = {self.k}, metric = '{self.metric}')")
		bnn = cg.BalancedKNN(k=self.k, metric=self.metric, maxl=2 * self.k, sight_k=2 * self.k, n_jobs=-1)
		bnn.fit(data)
		knn = bnn.kneighbors_graph(mode='distance')
		knn.eliminate_zeros()
		mknn = knn.minimum(knn.transpose())
		# Convert distances to similarities
		max_d = knn.data.max()
		knn.data = (max_d - knn.data) / max_d
		mknn.data = (max_d - mknn.data) / max_d
		mknn = mknn.tocoo()
		mknn.setdiag(0)
		# Compute the effective resolution
		d = 1 - knn.data
		radius = np.percentile(d, 90)
		logging.info(f"  90th percentile radius: {radius:.02}")
		inside = mknn.data > 1 - radius
		rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)

		return (radius, rnn.row, rnn.col, rnn.data)
