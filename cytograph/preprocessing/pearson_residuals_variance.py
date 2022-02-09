import shoji
import numpy as np
from cytograph import requires, creates, Algorithm
from ..utils import div0
import logging


class PearsonResidualsVariance(Algorithm):
	def __init__(self, **kwargs) -> None:
		"""
		Calculate the variance of the Pearson residuals for each gene

		Creates:
			PearsonResidualsVariance		Residuals variance per gene across cells

		Remarks:
			See equation 9 on p. 4 of https://doi.org/10.1101/2020.12.01.405886
		"""
		super().__init__(**kwargs)

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("OverallTotalUMIs", "uint64", ())
	@creates("PearsonResidualsVariance", "float32", ("genes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		logging.info(" PearsonResidualsVariance: Computing the variance of Pearson residuals")
		totals = self.TotalUMIs[:].astype("float32")
		gene_totals = self.GeneTotalUMIs[:].astype("float32")
		batch_size = 1000
		ix = 0
		n_cells = ws.cells.length
		acc = shoji.Accumulator()
		while ix < n_cells:
			data = self.Expression[ix: ix + batch_size, :]
			expected = totals[ix: ix + batch_size, None] @ div0(gene_totals[None, :], self.OverallTotalUMIs[:])
			residuals = div0((data - expected), np.sqrt(expected + np.power(expected, 2) / 100))
			residuals = np.clip(residuals, -np.sqrt(n_cells), np.sqrt(n_cells))
			for j in range(residuals.shape[0]):
				acc.add(residuals[j, :])
			ix += batch_size
		return acc.variance
