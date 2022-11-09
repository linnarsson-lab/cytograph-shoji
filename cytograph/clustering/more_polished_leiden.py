import logging
from typing import Tuple
import leidenalg as la
import igraph
import numpy as np
from cytograph import requires, creates, Algorithm
import shoji
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier


class MorePolishedLeiden(Algorithm):
	"""
	Leiden clustering with some post-processing
	"""
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

			Next, a classifier is trained on all clusters larger than `min_size`, and
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


	@requires("Embedding", "float32", ("cells", 2))
	@requires("Factors", "float32", ("cells", None))
	@requires("ManifoldIndices", "uint32", (None, 2))
	@requires("ManifoldWeights", "float32", (None))
	@creates("Clusters", "uint32", ("cells",))
	@creates("ClustersSecondary", "uint32", ("cells",))
	@creates("ClustersProbability", "float32", ("cells",))
	@creates("ClustersSecondaryProbability", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Given a sparse adjacency matrix, perform Leiden clustering

		Args:
			ws		The shoji Workspace

		Returns:
			labels:	The cluster labels
		"""
		n_cells = ws.cells.length
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

		# Assign each orphan cell to the same cluster as the nearest non-orphan
		logging.info(f" MorePolishedLeiden: Removing clusters with less than {self.min_size} cells")
		too_small = np.isin(labels, np.where(np.bincount(labels) < self.min_size)[0])
		# Relabel, in case some labels are missing
		labels_not_too_small = LabelEncoder().fit_transform(labels[~too_small])
		n_large_clusters = np.unique(labels_not_too_small).shape[0]
		logging.info(f" MorePolishedLeiden: {too_small.sum()} cells lost their cluster labels ({int(too_small.sum() / n_cells * 100)}%)")

		logging.info(f" MorePolishedLeiden: Reclassifying all cells to the remaining {n_large_clusters} clusters")
		factors = self.Factors[:]
		sgdc = SGDClassifier(loss="hinge", penalty="l2", class_weight='balanced', tol=0.01, n_jobs=-1)
		classifier = make_pipeline(StandardScaler(), CalibratedClassifierCV(sgdc))
		classifier.fit(factors[~too_small, :], labels_not_too_small)

		# Avoid materializing a whole probability matrix (n_cells, n_clusters), which might be too large
		ix = 0
		BATCH_SIZE = 10_000
		predicted = np.zeros(n_cells, dtype="uint32")
		secondary = np.zeros(n_cells, dtype="uint32")
		predicted_proba = np.zeros(n_cells, dtype="float32")
		secondary_proba = np.zeros(n_cells, dtype="float32")
		while ix < n_cells:
			probs = classifier.predict_proba(factors[ix: ix + BATCH_SIZE])
			ordered = probs.argsort(axis=1)
			predicted[ix: ix + BATCH_SIZE] = classifier.classes_[ordered[:, -1]]
			secondary[ix: ix + BATCH_SIZE] = classifier.classes_[ordered[:, -2]]
			predicted_proba[ix: ix + BATCH_SIZE] = probs[np.arange(len(ordered)), ordered[:, -1]]
			secondary_proba[ix: ix + BATCH_SIZE] = probs[np.arange(len(ordered)), ordered[:, -2]]
			ix += BATCH_SIZE
		
		labels[too_small] = predicted[too_small]  # New labels for the too_small clusters
		labels[~too_small] = labels_not_too_small  # Keep the labels for the not too small clusters, to avoid holes in the label sequence
		assert len(np.unique(labels)) == labels.max() + 1, "Missing cluster labels due to reclassification"

		accuracy = (predicted[~too_small] == labels[~too_small]).sum() / (~too_small).sum()
		logging.info(f" MorePolishedLeiden: {int(accuracy * 100)}% classification accuracy on non-orphan cells")

		return labels, secondary, predicted_proba, secondary_proba
