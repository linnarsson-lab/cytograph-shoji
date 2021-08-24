from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from ..utils import div0
from ..module import creates, requires, Module
import shoji
import logging


class ResidualsPCA(Module):
	"""
	Project a dataset into a reduced feature space using PCA on Pearson residuals.
	See https://www.biorxiv.org/content/10.1101/2020.12.01.405886v1.full.pdf
	"""
	def __init__(self, n_factors: int = 50, **kwargs) -> None:
		"""
		Args:
			n_factors:  	The number of retained components
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
		data = ws[ws.SelectedFeatures == True][self.requires["Expression"]]  # self.requires["Expression"] ensures that the user can rename the input tensor if desired
		expected = totals[:, None] @ (gene_totals[None, :] / self.OverallTotalUMIs[:])
		residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
		residuals = np.clip(residuals, -np.sqrt(n_cells), np.sqrt(n_cells))

		logging.info(f" ResidualsPCA: Computing principal components")
		pca = PCA(n_components=self.n_factors)
		factors = pca.fit_transform(residuals)
		loadings = pca.components_.T
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, self.n_factors))
		loadings_all[ws.SelectedFeatures[:]] = loadings
		return factors, loadings_all
