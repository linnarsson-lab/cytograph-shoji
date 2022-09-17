from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from ..utils import div0
from ..algorithm import creates, requires, Algorithm
import shoji
import logging


class ResidualsPCA(Algorithm):
	"""
	Project a dataset into a reduced feature space using PCA on Pearson residuals.
	See https://www.biorxiv.org/content/10.1101/2020.12.01.405886v1.full.pdf
	"""
	def __init__(self, n_factors: int = 50, **kwargs) -> None:
		"""
		Args:
			n_factors:  	The number of retained components

		Remarks:
			This algorithm loads all the data for selected features, which may require 
			a large amount of memory
		"""
		super().__init__(**kwargs)
		self.n_factors = n_factors

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("OverallTotalUMIs", "uint64", ())
	@creates("Factors", "float32", ("cells", None))
	@creates("Loadings", "float32", ("genes", None))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		logging.info(" ResidualsPCA: Computing Pearson residuals")
		n_cells = ws.cells.length
		totals = self.TotalUMIs[:].astype("float32")
		gene_totals = self.GeneTotalUMIs[ws.SelectedFeatures == True][:].astype("float32")
		overall_totals = totals.sum()
		data = ws[ws.SelectedFeatures == True][self.requires["Expression"]]  # self.requires["Expression"] ensures that the user can rename the input tensor if desired
		expected = totals[:, None] @ (gene_totals[None, :] / overall_totals)
		residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
		residuals = np.clip(residuals, 0, np.sqrt(n_cells))

		logging.info(f" ResidualsPCA: Computing principal components")
		pca = PCA(n_components=self.n_factors)
		factors = pca.fit_transform(residuals)
		loadings = pca.components_.T

		evs = ", ".join([f"{x:.2f}" for x in pca.explained_variance_ratio_ if x > 0.01]) + ", ..."
		logging.info(f" ResidualsPCA: Explained variance ({int(pca.explained_variance_ratio_.sum() * 100)}%): {evs}")

		keep_factors = self.n_factors
		if pca.explained_variance_ratio_.sum() > 0.5:
			keep_factors = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.5)[0])
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, keep_factors))
		loadings_all[ws.SelectedFeatures[:]] = loadings[:, :keep_factors]
		factors = factors[:, :keep_factors]
		return factors, loadings_all
