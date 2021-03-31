import logging
import shoji
import numpy as np
from cytograph import creates, requires, Module
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
import fastcluster


class Dendrogram(Module):
	def __init__(self, **kwargs) -> None:
		"""
		Compute a dendrogram of clusters

		Remarks:
			Agglomeration using Ward's linkage on the normalized mean expression per cluster.
			Mean expression values are normalized to the median total UMIs, and `log2(x + 1)` transformed.

			The returned Clusters and ClusterID are renumbered and correspond to the linkage ordering.
		"""
		super().__init__(**kwargs)

	@requires("Clusters", "uint32", ("cells",))
	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@creates("Linkage", "float32", (None, 4))
	@creates("ClusterID", "uint32", ("clusters",))
	@creates("Clusters", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool) -> np.ndarray:
		logging.info(" Dendrogram: Computing normalized expression of selected genes")
		selected = self.SelectedFeatures[:]
		x = self.MeanExpression[:]
		totals = x.sum(axis=1)
		x = np.log2((x.T / totals * np.median(totals)).T + 1)[:, selected]

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
		clusters = self.Clusters[:]
		cluster_ids = np.zeros(ws.clusters.length, dtype="uint32")
		new_clusters = np.zeros(ws.cells.length, dtype="uint32")
		for i in range(ws.clusters.length):
			selection = clusters == ordering[i]
			new_clusters[selection] = i
			cluster_ids[ordering[i]] = i

		return Z, cluster_ids, new_clusters
