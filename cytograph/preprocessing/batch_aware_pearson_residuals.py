from typing import Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder
from ..algorithm import creates, requires, Algorithm
from ..utils import div0
import shoji
import logging


class BatchAwarePearsonResiduals(Algorithm):
	"""
	Compute batch-aware Pearson residuals and their variance.
	See https://www.biorxiv.org/content/10.1101/2020.12.01.405886v1.full.pdf
	"""
	def __init__(self, batch_key: str = None, **kwargs) -> None:
		"""
		Args:
			batch_key:    The name of the tensor to use as batch key

		Remarks:
			This algorithm loads and saves data incrementally, in batches.

			Computes batch-aware Pearson residuals, where the gene totals are taken
			separately for each batch
		"""
		super().__init__(**kwargs)
		self.batch_key = batch_key

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("OverallTotalUMIs", "uint64", ())
	@creates("PearsonResidualsVariance", "float32", ("genes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		logging.info(" PearsonResiduals: Computing gene and cell statistics per batch")

		n_cells = ws.cells.length
		n_genes = ws.genes.length
		cell_totals = self.TotalUMIs[:].astype("float32")
		overall_totals = self.OverallTotalUMIs[:]

		stats = ws.cells.groupby(self.batch_key).stats(self.requires["Expression"])
		gene_totals = stats.sum()[1]  # shape (n_batches, n_genes)

		batch_size = 10_000
		logging.info(f" PearsonResiduals: Computing Pearson residuals incrementally in batches of {batch_size:,} cells")
		ws.PearsonResidualsTemp = shoji.Tensor(dtype="float32", dims=(None, "genes"), chunks=(32, 100))
		ws.PearsonResiduals = shoji.Tensor(dtype="float32", dims=(None, "genes"), chunks=(32, 100))
		if self.batch_key is not None:
			keys = LabelEncoder().fit_transform(ws[self.batch_key])
		else:
			keys = np.zeros(n_cells)
		unique_keys = np.unique(keys)
		acc = shoji.Accumulator()
		for ix in range(0, n_cells, batch_size):
			residuals = np.zeros((min(batch_size, n_cells - ix), n_genes), dtype="float32")
			data = self.Expression[ix:ix + batch_size]
			for j, key in enumerate(unique_keys):
				indices = (keys == key)[ix: ix + batch_size]
				if indices.sum() == 0:
					continue
				jdata = data[indices]
				expected = cell_totals[ix:ix + batch_size, None][indices] @ (gene_totals[j, :] / overall_totals)[None, :]
				residuals[indices, :] = div0((jdata - expected), np.sqrt(expected + np.power(expected, 2) / 100))
			residuals = np.clip(residuals, 0, np.sqrt(n_cells))
			ws.PearsonResidualsTemp.append(residuals)
		# Now rescale the residuals for each batch
		logging.info(" PearsonResiduals: Rescaling residuals per batch and computing variance")
		stats = ws.cells.groupby(self.batch_key).stats("PearsonResidualsTemp")
		means = stats.mean()[1]  # shape (n_batches, n_genes)
		size_factors = div0(means.mean(), means).astype("float32")
		for ix in range(0, n_cells, batch_size):
			residuals = ws.PearsonResidualsTemp[ix: ix + batch_size]
			for j, key in enumerate(unique_keys):
				indices = (keys == key)[ix: ix + batch_size]
				if indices.sum() == 0:
					continue
				residuals[indices, :] *= size_factors[j, :]
			ws.PearsonResiduals.append(residuals)
			for j in range(residuals.shape[0]):
				acc.add(residuals[j, :])
#			del ws.PearsonResidualsTemp
		return acc.variance
