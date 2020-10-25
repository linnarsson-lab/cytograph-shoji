import numpy as np
from sklearn.svm import SVR
from cytograph import requires, creates
import shoji
import logging


class FeatureSelectionByFeatureClustering:
	def __init__(self, n_genes: int) -> None:
		"""
		Select a subset of already selected features, by clustering features and selecting
		genes proportional to the inverse cluster size

		Args:
			n_genes		Number of genes to select
		"""
		self.n_genes = n_genes

	@requires("Expression", None, ("cells", "genes"))
	@requires("SelectedFeatures", "bool", ("genes",))
	@creates("SelectedFeatures", "bool", ("genes",), indices=True)
	def fit(self, ws: shoji.WorkspaceManager) -> np.ndarray:
		"""
		Clusters the previously selected features and selects a subset

		Args:
			ws:	shoji.Workspace containing the data to be used

		Returns:
			ndarray of indices of selected genes
		"""
		n_genes = ws.genes.length
		n_cells = ws.cells.length

		logging.info("FeatureSelectionByFeatureClustering: Loading data")
		x = ws[ws.SelectedFeatures == True].Expression
