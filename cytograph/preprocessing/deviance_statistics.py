import shoji
import numpy as np
from cytograph import requires, creates, Module
from ..utils import div0
import logging


class DevianceStatistics(Module):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

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
			See equation D_j on p. 14 of https://doi.org/10.1186/s13059-019-1861-6
		"""
		# Create symbolic names for the required tensors, which might be renamed by the user
		logging.info(" DevianceStatistics: Computing binomial deviance statistics for genes")
		n = self.TotalUMIs[...]
		n_sum = n.sum()
		gn = self.GeneTotalUMIs[...]
		d_j = np.zeros((1, ws.genes.length))
		for ix in range(0, ws.cells.length, 1000):
			pi_hat_j = (gn / n_sum)[None, :]
			y_ji = self.Expression[ix: ix + 1000]
			n_i = n[ix: ix + 1000][:, None]
			with np.errstate(divide='ignore', invalid='ignore'):
				# This would be faster as a numba nested for loop
				d_j += 2 * np.sum(np.nan_to_num(y_ji * np.log(div0(y_ji, (n_i * pi_hat_j)))) + (n_i - y_ji) * np.log((n_i - y_ji) / (n_i * (1 - pi_hat_j))), axis=0)
		return d_j[0]
