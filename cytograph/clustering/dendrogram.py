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
		"""
		super().__init__(**kwargs)

	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@creates("Linkage", "float32", (None, 4))
	@creates("LinkageOrdering", "uint32", ("clusters",))
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

		return Z, ordering
