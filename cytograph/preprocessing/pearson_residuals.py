from typing import Tuple

import numpy as np
from ..algorithm import creates, requires, Algorithm
import shoji
import logging
from ..utils import div0


class PearsonResiduals(Algorithm):
	"""
	Compute the clipped Pearson residuals.
	See https://www.biorxiv.org/content/10.1101/2020.12.01.405886v1.full.pdf
	"""
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("OverallTotalUMIs", "uint64", ())
	@creates("PearsonResiduals", "float32", ("cells", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		logging.info(" PearsonResiduals: Loading gene and cell totals")
		n_cells = ws.cells.length
		totals = self.TotalUMIs[:].astype("float32")
		gene_totals = self.GeneTotalUMIs[:].astype("float32")
		overall_totals = self.OverallTotalUMIs[:]

		logging.info(f" PearsonResiduals: Computing Pearson residuals")
		batch_size = 1000
		ix = 0
		n_cells = ws.cells.length
		n_genes = ws.genes.length
		residuals = np.empty((n_cells, n_genes), dtype="float32")
		while ix < n_cells:
			data = self.Expression[ix: ix + batch_size, :]
			expected = totals[ix: ix + batch_size, None] @ div0(gene_totals[None, :], overall_totals)
			residuals[ix: ix + batch_size, :] = div0((data - expected), np.sqrt(expected + np.power(expected, 2) / 100))
			residuals[ix: ix + batch_size, :] = np.clip(residuals[ix: ix + batch_size, :], -np.sqrt(n_cells), np.sqrt(n_cells))
			ix += batch_size

		return residuals
