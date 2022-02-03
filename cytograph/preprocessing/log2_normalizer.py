from typing import Tuple
import numpy as np
import shoji
from ..utils import div0
from cytograph import requires, creates, Module
import logging


class Log2Normalizer(Module):
	"""
	Normalize and log2-transform a dataset, dealing properly
	with edge cases such as division by zero.
	"""
	def __init__(self, level: int = 0, **kwargs) -> None:
		"""
		Normalize and log2-transform a dataset.
		
		Args:
			level			The level to which each cell should be normalized (or 0 to use the median)

		Creates:
			Log2Level:		The level to which all cells should be normalized
			Log2Mean: 		The mean of the log2-transformed values

		Remarks:
			The normalization is not saved to the workspace. Instead, Log2Level and Log2Mean
			are saved, which can be used later to transform expression data. To transform
			new data using an existing transform, use Log2Normalizer.load()
		"""
		super().__init__(**kwargs)
		self.log2_mu = None  # type: np.ndarray
		self.totals = None  # type: np.ndarray
		self.level = level

	@requires("Expression", None, ("cells", "genes"))
	@requires("TotalUMIs", "uint32", ("cells",))
	@creates("Log2Level", "uint16", ())
	@creates("Log2Mean", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[int, np.ndarray]:
		logging.info(" Log2Normalizer: Computing normalized log2 mean")
		n_genes = ws.genes.length
		n_cells = ws.cells.length
		self.log2_mu = np.zeros(n_genes)
		self.totals = self.TotalUMIs[:]
		if self.level == 0:
			self.level = np.median(self.totals)

		for ix in range(0, n_cells, 1000):
			vals = self.Expression[ix: ix + 1000].astype("float")
			# Rescale to the median total UMI count, plus 1 (to avoid log of zero), then log transform
			vals = np.log2(div0(vals.T, self.totals) * self.level + 1).T
			self.log2_mu[ix: ix + 1000] = np.mean(vals, axis=1)
		return (self.level, self.log2_mu)

	@classmethod
	def load(ws: shoji.WorkspaceManager) -> "Log2Normalizer":
		"""
		Load a previoulsy stored normalization from the workspace
		"""
		nn = Log2Normalizer(ws.Log2Level[:])
		nn.totals = ws.TotalUMIs[:]
		nn.log2_mu = ws.Log2Mean[:]
		return nn

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
