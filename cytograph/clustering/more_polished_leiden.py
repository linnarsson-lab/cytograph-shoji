import logging
from typing import Tuple
import leidenalg as la
import igraph
import numpy as np
from cytograph import requires, creates, Module
import shoji
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class MorePolishedLeiden(Module):
	def __init__(self, resolution: float = 1.0, method: str = "modularity", max_size: int = 0, min_size: int = 10, **kwargs) -> None:
		"""
		Find clusters on the manifold using the Leiden algorithm, then polish clusters on the embedding.

		Args:
			resolution: The resolution parameter (typically 0.01 - 1; default: 1)
			method:     The partitioning method ("modularity", "cpm", "surprise", "rb", "rber", or "significance"; default: "modularity")
			max_size:   The maximum size of clusters (default: 0, i.e. no limit)
			min_size:   The minimum size of clusters (default: 10)

		Remarks:
			The default method, "modularity", is equivalent to Louvain clustering but the Leiden algorithm is faster
			and can yield better clusters. The method "surprise" can yield clusters similar to Amos Tanay's Metacell
			algorithm (i.e. a tiling of the manifold with clusters of similar size), especially if `max_size` is used
			to cap the cluster size.

			The polishing step consists of two phases. First, the shape of each cluster on the embedding (e.g. TSNE) is
			considered using Iglewicz-Hoaglin outlier detection, and clusters with too many outliers are re-clustered
			using DBSCAN on the embedding. This step is identical to standard PolishedLeiden. 
			
			Second, a classifier is trained on all clusters larger than `min_size`, and
			all cells are reassigned to their maximum probability cluster. This removes all small clusters and outliers.
			Note that this can result in clusters larger than `max_size`. The probability of the cluster label for
			each cell is returned, and can be used to identify outliers or intermediate cell states.

			The resolution parameter has no effect on the default method, "modularity".
		"""
		super().__init__(**kwargs)
		self.resolution = resolution
		ptypes = {
			"modularity": la.ModularityVertexPartition,
			"cpm": la.CPMVertexPartition,
			"surprise": la.SurpriseVertexPartition,
			"rb": la.RBConfigurationVertexPartition,
			"rber": la.RBERVertexPartition,
			"significance": la.SignificanceVertexPartition
		}
		if method.lower() in ptypes:
			self.method = ptypes[method.lower()]
		else:
			raise ValueError(f"Invalid partition method '{method}'")
		self.max_size = max_size
		self.min_size = min_size

	def _is_outlier(self, points: np.ndarray, thresh: float = 3.5) -> np.ndarray:
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
			outliers[~outliers] = self._is_outlier(x[~outliers])
			outliers[~outliers] = self._is_outlier(y[~outliers])

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
		labels = clusterer.fit_predict(xy)

		# Assign each outlier to the same cluster as the nearest non-outlier
		if (labels == -1).sum() > 0:
			nn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
			nn.fit(xy[labels >= 0])
			nearest = nn.kneighbors(xy[labels == -1], n_neighbors=1, return_distance=False)
			labels[labels == -1] = labels[labels >= 0][nearest.flat[:]]
		return labels

	@requires("Embedding", "float32", ("cells", 2))
	@requires("Factors", "float32", ("cells", None))
	@requires("ManifoldIndices", "uint32", (None, 2))
	@requires("ManifoldWeights", "float32", (None))
	@creates("Clusters", "uint32", ("cells",))
	@creates("ClustersSecondary", "uint32", ("cells",))
	@creates("ClusterProbability", "float32", ("cells",))
	@creates("ClusterSecondaryProbability", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Given a sparse adjacency matrix, perform Leiden clustering

		Args:
			ws		The shoji Workspace

		Returns:
			labels:	The cluster labels
		"""
		logging.info(" MorePolishedLeiden: Loading graph data")
		rc = self.ManifoldIndices[:]
		logging.info(" MorePolishedLeiden: Constructing the graph")
		weights = self.ManifoldWeights[:]
		g = igraph.Graph(ws.cells.length, list(zip(rc[:, 0].T.tolist(), rc[:, 1].T.tolist())), directed=False, edge_attrs={'weight': weights})
		logging.info(" MorePolishedLeiden: Optimizing the graph partitioning")
		if self.resolution != 1:
			labels = np.array(la.find_partition(g, self.method, weights=weights, max_comm_size=self.max_size, resolution_parameter=self.resolution, n_iterations=-1).membership)
		else:
			labels = np.array(la.find_partition(g, self.method, weights=weights, max_comm_size=self.max_size, n_iterations=-1).membership)
		logging.info(f" MorePolishedLeiden: Found {labels.max() + 1} initial clusters")

		# Break clusters based on the embedding
		logging.info(" MorePolishedLeiden: Breaking clusters based on the embedding")
		xy = self.Embedding[:]
		# Only break clusters that are at least twice as large as the minimum size (note: labels are sorted by cluster size)
		max_label = np.where(np.bincount(labels) < self.min_size * 2)[0][0]
		next_label = 0
		labels2 = np.copy(labels)
		for lbl in range(max_label):
			cluster = labels == lbl
			if cluster.sum() < self.min_size:
				continue
			adjusted = self._break_cluster(xy[cluster, :])
			new_labels = np.copy(adjusted)
			for i in range(np.max(adjusted) + 1):
				new_labels[adjusted == i] = i + next_label
			next_label = next_label + np.max(adjusted) + 1
			labels2[cluster] = new_labels
		labels = labels2
		logging.info(f" MorePolishedLeiden: Found {labels.max() + 1} clusters after breaking clusters on the embedding")

		# Assign each orphan cell to the same cluster as the nearest non-orphan
		logging.info(f" MorePolishedLeiden: Removing clusters with less than {self.min_size} cells")
		too_small = np.isin(labels, np.where(np.bincount(labels) < self.min_size)[0])
		n_large_clusters = np.unique(labels[~too_small]).shape[0]
		logging.info(f" MorePolishedLeiden: {too_small.sum()} ({int(too_small.sum() / too_small.shape[0] * 100)}%) cells lost their cluster labels")

		logging.info(f" MorePolishedLeiden: Reclassifying all cells to the remaining {n_large_clusters} clusters")
		factors = self.Factors[:]
		classifier = make_pipeline(StandardScaler(), RandomForestClassifier(oob_score=True, class_weight='balanced', n_jobs=-1))
		classifier.fit(factors[~too_small, :], labels[~too_small])
		probs = classifier.predict_proba(factors)
		oob_score = classifier.named_steps["randomforestclassifier"].oob_score_
		logging.info(f" MorePolishedLeiden: Out-of-band score {oob_score:.2f}")
		
		# We would really like to renumber the clusters, but the 'secondary' calculation below makes it difficult
		# predicted = probs.argmax(axis=1)
		# labels = LabelEncoder().fit_transform(predicted)  # Renumber just in case some cluster gets zero cells
		# max_proba = probs[np.arange(len(labels)), labels]

		ordered = probs.argsort(axis=1)
		predicted = ordered[:, -1]
		secondary = ordered[:, -2]
		max_proba = probs[np.arange(len(predicted)), predicted]
		secondary_proba = probs[np.arange(len(secondary)), secondary]
		assert len(np.unique(predicted)) == predicted.max() + 1, "Missing cluster labels due to reclassification"
		return predicted, secondary, max_proba, secondary_proba
