from typing import Tuple
import shoji
import numpy as np
import cytograph as cg
from cytograph import requires, creates
import logging


class DevianceStatistics:
	def __init__(self) -> None:
		pass

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@creates("Deviance", "float32", ("genes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		"""
		Calculate binomial deviance statistics for each gene

		Args:
			ws				shoji workspace
			save			if true, save the result to the workspace

		Returns:
			Deviance		Deviance per gene

		Remarks:
			If the tensor "ValidGenes" exists, only ValidGenes == True genes will be selected
			See equation D_j on p. 14 of https://doi.org/10.1186/s13059-019-1861-6
		"""
		logging.info(" DevianceStatistics: Computing binomial deviance statistics for genes")
		n = ws[:].TotalUMIs
		gn = ws[:].GeneTotalUMIs
		pi_hat_j = gn / n.sum()
		d = 0
		for ix in range(0, ws.cells.length, 1000):
			y_ij = ws.Expression[ix: ix + 1000]
			n_i = n[ix: ix + 1000]
			d += 2 * (y_ij * np.log(y_ij / (n_i * pi_hat_j)) + (n_i - y_ij) * np.log((n_i - y_ij) / (n_i * (1 - pi_hat_j))))
		return d
