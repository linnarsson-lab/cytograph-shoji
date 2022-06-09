import logging
import shoji
import numpy as np
from cytograph import creates, requires, Algorithm
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
import fastcluster


def indices_to_order_a_like_b(a, b):
	return a.argsort()[b.argsort().argsort()]


class Dendrogram(Algorithm):
	"""
	Compute a dendrogram of clusters
	"""
	def __init__(self, **kwargs) -> None:
		"""
		Compute a dendrogram of clusters

		Remarks:
			Agglomeration using Ward's linkage on the normalized mean expression per cluster.
			Mean expression values are normalized to the median total UMIs, and `log2(x + 1)` transformed.

			The returned Clusters and ClusterID are renumbered and correspond to the linkage ordering.

			If the tensors are saved to the workspace (save == True), then all the aggregated tensors
			(i.e. those on the 'clusters' dimension) are also reordered in cluster order.
		"""
		super().__init__(**kwargs)

	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@creates("Linkage", "float32", (None, 4))
	@creates("ClusterID", "uint32", ("clusters",))
	@creates("Clusters", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		logging.info(" Dendrogram: Computing normalized expression of selected genes")
		selected = self.SelectedFeatures[:]
		x = self.MeanExpression[:]
		totals = x.sum(axis=1)
		x = np.log2((x.T / totals * np.median(totals)).T + 1)[:, selected]
		x = scale(x)

		logging.info(" Dendrogram: Computing Ward's linkage")
		D = pdist(x, 'correlation')
		Z = fastcluster.linkage(D, 'ward', preserve_input=True)
		Z = hc.optimal_leaf_ordering(Z, D)
		ordering = hc.leaves_list(Z)

		# Renumber the linkage matrix so it corresponds to the reordered clusters
		for i in range(len(Z)):
			if Z[i, 0] < x.shape[0]:
				Z[i, 0] = np.where(Z[i, 0] == ordering)[0]
			if Z[i, 1] < x.shape[0]:
				Z[i, 1] = np.where(Z[i, 1] == ordering)[0]

		# Renumber the clusters according to the new ordering
		cluster_ids = np.zeros(ws.clusters.length, dtype="uint32")
		new_clusters = np.zeros(ws.cells.length, dtype="uint32")
		for i in range(ws.clusters.length):
			cluster_ids[ordering[i]] = i
		clusters = self.Clusters[:]
		old_cluster_ids = self.ClusterID[:]
		for i, j in zip(old_cluster_ids, cluster_ids):
			new_clusters[clusters == i] = j

		if save:
			# Reorder the aggregated tensors to match the numbering
			ordering = np.argsort(cluster_ids)
			logging.info(" Dendrogram: Reordering aggregated tensors")
			for tname in ws._tensors():
				tensor = ws[tname]
				if tensor.rank >= 1 and tensor.dims[0] == "clusters":
					ws[tname] = shoji.Tensor(tensor.dtype, tensor.dims, chunks=tensor.chunks, inits=tensor[:][ordering])

			return Z, cluster_ids[ordering], new_clusters
		else:
			return Z, cluster_ids, new_clusters
