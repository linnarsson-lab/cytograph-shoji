from typing import List, Dict
import sys
import shoji
import numpy as np
import logging


class InitializeWorkspace:
	def __init__(self, from_workspace: str, tensors: List[str]) -> None:
		"""
		Args:
			from_workspace	Full shoji path of the workspace to use as template
			tensors			List of names of tensors to import from the template workspace
		
		Remarks:
			The new workspace will be initialized by creating
			cells and genes dimensions, and importing all the tensors listed (supports only scalars and gene tensors)
		"""
		self.from_workspace = from_workspace
		self.tensors = tensors

	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		"""
		Args:
			ws		shoji workspace
		"""
		db = shoji.connect()
		ws.genes = shoji.Dimension(shape=None)
		ws.cells = shoji.Dimension(shape=None)

		logging.info(f" InitializeWorkspace: Collecting tensors from '{self.from_workspace}'")
		d: Dict[str, np.ndarray] = {}
		for tensor in self.tensors:
			if tensor not in db[self.from_workspace]:
				logging.error(f"Tensor '{tensor}' was not found in workspace '{self.from_workspace}'")
				sys.exit(1)
			t = db[self.from_workspace][tensor]
			if t.rank > 0 and t.dims[0] == "genes":
				ws[tensor] = shoji.Tensor(dtype=t.dtype, dims=t.dims)
				d[tensor] = t[:]
			elif t.rank == 0:
				ws[tensor] = shoji.Tensor(dtype=t.dtype, dims=t.dims, inits=t[:])
			else:
				raise ValueError(f"InitializeWorkspace can only import scalars and tensors along 'genes' dimension, but '{tensor}' was rank-{t.rank} and first dimension was '{t.dims[0]}'")
		ws.genes.append(d)
		ws.genes = shoji.Dimension(shape=ws.genes.length)  # Fix the length of the genes dimension
