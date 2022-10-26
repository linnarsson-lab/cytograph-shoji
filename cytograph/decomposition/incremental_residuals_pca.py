from typing import Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
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
			data = self.Expression[ix:ix + batch_size, ws.SelectedFeatures == True]
			expected = totals[ix:ix + batch_size, None] @ (gene_totals[None, :] / overall_totals)
			residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
			residuals = np.clip(residuals, 0, np.sqrt(n_cells))
#			residuals = np.log2(residuals + 1)
			pca.partial_fit(residuals)

		logging.info(f" ResidualsPCA: Transforming residuals incrementally in batches of {batch_size:,} cells")
		factors = np.zeros((ws.cells.length, self.n_factors), dtype="float32")
		for ix in range(0, ws.cells.length, batch_size):
			data = self.Expression[ix:ix + batch_size, ws.SelectedFeatures == True]
			expected = totals[ix:ix + batch_size, None] @ (gene_totals[None, :] / overall_totals)
			residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
			residuals = np.clip(residuals, 0, np.sqrt(n_cells))
#			residuals = np.log2(residuals + 1)
			factors[ix:ix + batch_size] = pca.transform(residuals)

		evs = ", ".join([f"{x:.2f}" for x in pca.explained_variance_ratio_ if x > 0.01]) + ", ..."
		logging.info(f" ResidualsPCA: Explained variance ({int(pca.explained_variance_ratio_.sum() * 100)}%): {evs}")

		keep_factors = self.n_factors
		if pca.explained_variance_ratio_.sum() > 0.5:
			logging.info("Keeping factors to 50% variance", keep_factors)
			keep_factors = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.5)[0])
			if keep_factors <= self.n_factors:
				keep_factors = self.n_factors
		loadings = pca.components_.T
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, keep_factors))
		loadings_all[ws.SelectedFeatures[:]] = loadings[:, :keep_factors]

		return factors[:, :keep_factors], loadings_all
