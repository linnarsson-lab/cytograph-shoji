import logging
import community
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from cytograph import creates
import scipy.sparse as sparse
import shoji


class Louvain:
	def __init__(self, resolution: float = 1.0, min_cells: int = 10, embedding: str = "TSNE") -> None:
		"""
		Args:
			resolution		The clustering resolution parameter
			min_cells		Minimum number of cells in a cluster
			embedding		Name of the embedding to use to eliminate outliers
		"""
		self.resolution = resolution
		self.min_cells = min_cells
		self.embedding = embedding

	@creates("Clusters", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		"""
		Given a sparse adjacency matrix, perform Louvain clustering, then polish the result

		Args:
			ws			The workspace
			save		If true, save the result to the workspace

		Returns:
			Clusters	The cluster labels

		"""
		if self.embedding in ws:
			xy = ws[self.embedding][:]
		else:
			raise ValueError(f"Embedding '{self.embedding}' not found in file")

		knn = sparse.coo_matrix((ws[:].RNN_data, (ws[:].RNN_row, ws[:].RNN_col)))

		logging.info("Louvain community detection")
		g = nx.from_scipy_sparse_matrix(knn)
		partitions = community.best_partition(g, resolution=self.resolution, randomize=False)
		labels = np.array([partitions[key] for key in range(knn.shape[0])])

		# Mark tiny clusters as outliers
		logging.info("Marking tiny clusters as outliers")
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
			logging.warn("All cells were determined to be outliers!")
			return np.zeros_like(labels)

		if np.any(labels == -1):
			# Assign each outlier to the same cluster as the nearest non-outlier
			nn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
			nn.fit(xy[labels >= 0])
			nearest = nn.kneighbors(xy[labels == -1], n_neighbors=1, return_distance=False)
			labels[labels == -1] = labels[labels >= 0][nearest.flat[:]]

		return labels
