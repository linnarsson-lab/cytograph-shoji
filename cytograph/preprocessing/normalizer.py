import numpy as np
import shoji
from .utils import div0
from cytograph import CytographMethod
import logging


class Log2Normalizer(CytographMethod):
	"""
	Normalize and log2-transform a dataset, dealing properly
	with edge cases such as division by zero.
	"""
	def __init__(self, level: int = None) -> None:
		"""
		Normalize and log2-transform a dataset.
		
		Args:
			level			The level to which each cell should be normalized (or None to use the median)
		
		Remarks:
			Requires the tensors TotalUMIs, MeanExpression and StdevExpression
		"""
		self._requires = [
			("Expression", None, ("cells", "genes")),
			("TotalUMIs", "uint32", ("cells",))
		]
		self.log2_mu = None  # type: np.ndarray
		self.totals = None  # type: np.ndarray
		self.level = level

	def fit(self, ws: shoji.WorkspaceManager) -> None:
		self.check(ws, "Log2Normalizer")
		logging.info("Log2Normalizer: Computing normalized log2 mean")
		n_genes = ws.genes.length
		n_cells = ws.cells.length
		self.log2_mu = np.zeros(n_genes)
		self.totals = ws[:].TotalUMIs
		if self.level == 0:
			self.level = np.median(self.totals)

		for ix in range(0, n_cells, 1000):
			vals = ws.Expression[ix: ix + 1000][:, :].astype("float")
			# Rescale to the median total UMI count, plus 1 (to avoid log of zero), then log transform
			vals = np.log2(div0(vals.T, self.totals) * self.level + 1).T
			self.log2_mu[ix: ix + 1000] = np.mean(vals, axis=1)
		logging.info("Log2Normalizer: Done.")

	def transform(self, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
		"""
		Normalize a matrix using the previously calculated aggregate statistics

		Args:
			vals (ndarray):		Matrix of shape (n_genes, n_cells)
			cells (ndarray):	Optional indices of the cells that are represented in vals

		Returns:
			vals_adjusted (ndarray):	The normalized values
		"""
		# Adjust total count per cell to the desired overall level
		if cells is None:
			cells = slice(None)
		vals = vals.astype("float")
		vals = np.log2(div0(vals, self.totals[cells]) * self.level + 1)

		# Subtract mean per gene
		vals = vals - self.log2_mu[:, None]
		return vals

	def fit_transform(self, ws: shoji.WorkspaceManager, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
		self.fit(ws)
		return self.transform(vals, cells)
