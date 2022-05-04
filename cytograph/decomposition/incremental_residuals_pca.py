from typing import Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA
from ..algorithm import creates, requires, Algorithm
import shoji
import logging


class IncrementalResidualsPCA(Algorithm):
	"""
	Project a dataset into a reduced feature space using PCA on Pearson residuals.
	See https://www.biorxiv.org/content/10.1101/2020.12.01.405886v1.full.pdf
	"""
	def __init__(self, n_factors: int = 50, **kwargs) -> None:
		"""
		Args:
			n_factors:  	The number of retained components

		Remarks:
			This algorithm loads data incrementally, in batches.
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
		logging.info(" ResidualsPCA: Loading gene and cell totals")
		n_cells = ws.cells.length
		self.n_factors = min(self.n_factors, n_cells)
		totals = self.TotalUMIs[:].astype("float32")
		gene_totals = self.GeneTotalUMIs[ws.SelectedFeatures == True][:].astype("float32")
		overall_totals = self.OverallTotalUMIs[:]

		batch_size = 100_000
		while n_cells % batch_size < self.n_factors:  # This is necessary to avoid a small final batch with too few samples
			batch_size -= self.n_factors
		logging.info(f" ResidualsPCA: Fitting PCA on Pearson residuals incrementally in batches of {batch_size:,} cells")
		pca = IncrementalPCA(n_components=self.n_factors)
		for ix in range(0, n_cells, batch_size):
			data = ws[self.requires["Expression"]][ix:ix + batch_size, ws.SelectedFeatures == True]  # self.requires["Expression"] ensures that the user can rename the input tensor if desired
			expected = totals[ix:ix + batch_size, None] @ (gene_totals[None, :] / overall_totals)
			residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
			residuals = np.clip(residuals, 0, np.sqrt(n_cells))
			pca.partial_fit(residuals)

		logging.info(f" ResidualsPCA: Transforming residuals incrementally in batches of {batch_size:,} cells")
		factors = np.zeros((ws.cells.length, self.n_factors), dtype="float32")
		for ix in range(0, ws.cells.length, batch_size):
			data = ws[self.requires["Expression"]][ix:ix + batch_size, ws.SelectedFeatures == True]  # self.requires["Expression"] ensures that the user can rename the input tensor if desired
			expected = totals[ix:ix + batch_size, None] @ (gene_totals[None, :] / overall_totals)
			residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
			residuals = np.clip(residuals, 0, np.sqrt(n_cells))
			factors[ix:ix + batch_size] = pca.transform(residuals)

		loadings = pca.components_.T
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, self.n_factors))
		loadings_all[ws.SelectedFeatures[:]] = loadings
		return factors, loadings_all
