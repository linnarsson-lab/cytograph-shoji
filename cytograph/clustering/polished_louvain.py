import logging

import community
import networkx as nx
import numpy as np
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
from cytograph import requires, creates
import shoji


def is_outlier(points: np.ndarray, thresh: float = 3.5) -> np.ndarray:
	"""
	Returns a boolean array with True if points are outliers and False
	otherwise.

	Parameters:
	-----------
		points : An numobservations by numdimensions array of observations
		thresh : The modified z-score to use as a threshold. Observations with
			a modified z-score (based on the median absolute deviation) greater
			than this value will be classified as outliers.

	Returns:
	--------
		mask : A numobservations-length boolean array.

	References:
	----------
		Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
		Handle Outliers", The ASQC Basic References in Quality Control:
		Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
	"""
	if len(points.shape) == 1:
		points = points[:, None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation

	return modified_z_score > thresh


class PolishedLouvain:
	def __init__(self, resolution: float = 1.0, min_cells: int = 10) -> None:
		self.resolution = resolution
		self.min_cells = min_cells

	def _break_cluster(self, embedding: np.ndarray) -> np.ndarray:
		"""
		If needed, split the cluster by density clustering on the embedding

		Returns:
			An array of cluster labels (all zeros if cluster wasn't split)
			Note: the returned array may contain -1 for outliers
		"""
		# Find outliers in either dimension using Grubbs test
		xy = PCA().fit_transform(embedding)
		x = xy[:, 0]
		y = xy[:, 1]
		# Standardize x and y (not sure if this is really necessary)
		x = (x - x.mean()) / x.std()
		y = (y - y.mean()) / y.std()
		xy = np.vstack([x, y]).transpose()

		outliers = np.zeros(embedding.shape[0], dtype='bool')
		for _ in range(5):
			outliers[~outliers] = is_outlier(x[~outliers])
			outliers[~outliers] = is_outlier(y[~outliers])

		# See if the cluster is very dispersed
		min_pts = min(50, min(x.shape[0] - 1, max(5, round(0.1 * x.shape[0]))))
		nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
		nn.fit(xy)
		knn = nn.kneighbors_graph(mode='distance')
		k_radius = knn.max(axis=1).toarray()
		epsilon = np.percentile(k_radius, 70)
		# Not too many outliers, and not too dispersed
		if outliers.sum() <= 3 and (np.sqrt(x**2 + y**2) < epsilon).sum() >= min_pts * 0.5:
			return np.zeros(embedding.shape[0], dtype='int')

		# Too many outliers, or too dispersed
		clusterer = DBSCAN(eps=epsilon, min_samples=round(min_pts * 0.5))
		return clusterer.fit_predict(xy)

	@requires("TSNE", "float32", ("cells", 2))
	@creates("Clusters", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		"""
		Given a sparse adjacency matrix, perform Louvain clustering, then polish the result

		Args:
			ws		The shoji Workspace

		Returns:
			labels:	The cluster labels
		"""
		xy = ws[:].TSNE
		knn = sparse.coo_matrix((ws[:].RNN_data, (ws[:].RNN_row, ws[:].RNN_col)))

		logging.info("PolishedLouvain: Louvain community detection")
		g = nx.from_scipy_sparse_matrix(knn)
		partitions = community.best_partition(g, resolution=self.resolution, randomize=False)
		labels = np.array([partitions[key] for key in range(knn.shape[0])])

		# Mark outliers using DBSCAN
		logging.info("PolishedLouvain: Using DBSCAN to mark outliers")
		nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", n_jobs=4)
		nn.fit(xy)
		knn = nn.kneighbors_graph(mode='distance')
		k_radius = knn.max(axis=1).toarray()
		epsilon = np.percentile(k_radius, 80)
		clusterer = DBSCAN(eps=epsilon, min_samples=10)
		outliers = (clusterer.fit_predict(xy) == -1)
		labels[outliers] = -1

		# Mark outliers as cells in bad neighborhoods
		logging.info("PolishedLouvain: Using neighborhood to mark outliers")
		nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", n_jobs=4)
		nn.fit(xy)
		knn = nn.kneighbors_graph(mode='connectivity').tocoo()
		temp = []
		for ix in range(labels.shape[0]):
			if labels[ix] == -1:
				temp.append(-1)
				continue
			neighbors = knn.col[np.where(knn.row == ix)[0]]
			neighborhood = labels[neighbors] == labels[ix]
			if neighborhood.sum() / neighborhood.shape[0] > 0.2:
				temp.append(labels[ix])
			else:
				temp.append(-1)
		labels = np.array(temp)

		# Renumber the clusters
		retain = sorted(list(set(labels)))
		d = dict(zip(retain, np.arange(-1, len(set(retain)))))
		labels = np.array([d[x] if x in d else -1 for x in labels])

		# Break clusters based on the embedding
		logging.info("PolishedLouvain: Breaking clusters")
		max_label = 0
		labels2 = np.copy(labels)
		for lbl in range(labels.max() + 1):
			cluster = labels == lbl
			if cluster.sum() < 10:
				continue
			adjusted = self._break_cluster(xy[cluster, :])
			new_labels = np.copy(adjusted)
			for i in range(np.max(adjusted) + 1):
				new_labels[adjusted == i] = i + max_label
			max_label = max_label + np.max(adjusted) + 1
			labels2[cluster] = new_labels
		labels = labels2

		# Set the local cluster label to the local majority vote
		logging.info("PolishedLouvain: Smoothing cluster identity on the embedding")
		nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", n_jobs=4)
		nn.fit(xy)
		knn = nn.kneighbors_graph(mode='connectivity').tocoo()
		temp = []
		for ix in range(labels.shape[0]):
			neighbors = knn.col[np.where(knn.row == ix)[0]]
			temp.append(mode(labels[neighbors])[0][0])
		labels = np.array(temp)

		# Mark tiny clusters as outliers
		logging.info("PolishedLouvain: Marking tiny clusters as outliers")
		ix, counts = np.unique(labels, return_counts=True)
		labels[np.isin(labels, ix[counts < self.min_cells])] = -1

		# Renumber the clusters (since some clusters might have been lost in poor neighborhoods)
		retain = list(set(labels))
		if -1 not in retain:
			retain.append(-1)
		retain = sorted(retain)
		d = dict(zip(retain, np.arange(-1, len(set(retain)))))
		labels = np.array([d[x] if x in d else -1 for x in labels])

		if np.all(labels < 0):
			logging.warn("PolishedLouvain: All cells were determined to be outliers!")
			return np.zeros_like(labels)

		if np.any(labels == -1):
			# Assign each outlier to the same cluster as the nearest non-outlier
			nn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
			nn.fit(xy[labels >= 0])
			nearest = nn.kneighbors(xy[labels == -1], n_neighbors=1, return_distance=False)
			labels[labels == -1] = labels[labels >= 0][nearest.flat[:]]

		return labels
