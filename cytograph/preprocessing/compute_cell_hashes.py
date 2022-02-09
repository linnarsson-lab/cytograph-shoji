import logging
from hashlib import sha1

import numpy as np
import shoji

from ..algorithm import Algorithm, creates, requires


class ComputeCellHashes(Algorithm):
	"""
	Compute invariant cell hashes based on the expression matrix. Cell hashes can be used to
	identify cells from different analysis runs.
	"""
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	@requires("Expression", "uint16", ("cells", "genes"))
	@creates("Hashes", "uint64", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		logging.info(" ComputeCellHashes: Loading expression data and computing hashes")
		n_cells = ws.cells.length

		hashes = np.zeros(n_cells, dtype="uint64")
		batch_size = 100_000
		for ix in range(0, n_cells, batch_size):
			x = self.Expression[ix:ix + batch_size, :2000]
			hashes[ix: ix + batch_size] = np.array([np.frombuffer(sha1(x[i, :].copy()).digest()[:8], dtype="uint64") for i in range(len(x))]).flatten()
		return hashes
