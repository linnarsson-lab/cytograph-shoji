import logging
import shoji
import numpy as np
from cytograph import creates, requires, Module
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
import fastcluster
import numpy_groupies as npg


class ResidualsDendrogram(Module):
	def __init__(self, **kwargs) -> None:
		"""
		Compute a dendrogram of clusters using Pearson residuals

		Remarks:
			Agglomeration using Ward's linkage on the mean Pearson residuals.
		"""
		super().__init__(**kwargs)

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("OverallTotalUMIs", "uint64", ())
	@requires("Clusters", "uint32", ("cells",))
	@creates("Linkage", "float32", (None, 4))
	@creates("LinkageOrdering", "uint32", ("clusters",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool) -> np.ndarray:
		logging.info(" ResidualsDendrogram: Computing Pearson residuals for selected genes")
		totals = self.TotalUMIs[:].astype("float32")
		gene_totals = self.GeneTotalUMIs[ws.SelectedFeatures == True][:].astype("float32")
		data = ws[ws.SelectedFeatures == True][self.requires["Expression"]]  # self.requires["Expression"] ensures that the user can rename the input tensor if desired
		expected = totals[:, None] @ (gene_totals[None, :] / self.OverallTotalUMIs[:])
		residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
		mean_residuals_by_cluster = npg.aggregate(self.Clusters[:], residuals, func='mean', fill_value=0, axis=1)

		logging.info(" ResidualsDendrogram: Computing Ward's linkage on mean residuals")
		D = pdist(mean_residuals_by_cluster, 'correlation')
		Z = fastcluster.linkage(D, 'ward', preserve_input=True)
		Z = hc.optimal_leaf_ordering(Z, D)
		ordering = hc.leaves_list(Z)

		return Z, ordering
