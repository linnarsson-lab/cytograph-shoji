from typing import List
import sys
import shoji
import logging
from cytograph import Module


class InitializeWorkspace(Module):
	def __init__(self, from_workspace: str, tensors: List[str], **kwargs) -> None:
		"""
		Args:
			from_workspace	Full shoji path of the workspace to use as template
			tensors			List of names of tensors to import from the template workspace
		
		Remarks:
			The new workspace will be initialized by creating
			cells and genes dimensions, and importing all the tensors listed (supports only scalars and gene tensors)
		"""
		super().__init__(**kwargs)

		self.from_workspace = from_workspace
		self.tensors = tensors

	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		"""
		Args:
			ws		shoji workspace
		"""
		db = shoji.connect()
		ws.cells = shoji.Dimension(shape=None)

		logging.info(f" InitializeWorkspace: Collecting tensors from '{self.from_workspace}'")
		for tensor in self.tensors:
			if tensor not in db[self.from_workspace]:
				logging.error(f"Tensor '{tensor}' was not found in workspace '{self.from_workspace}'")
				sys.exit(1)
			t = db[self.from_workspace][tensor]
			if t.dims[0] == "genes":
				if "genes" not in ws:
					ws.genes = shoji.Dimension(shape=t.shape[0])
				ws[tensor] = shoji.Tensor(dtype=t.dtype, dims=t.dims, inits=t[:])
			elif t.rank == 0:
				ws[tensor] = shoji.Tensor(dtype=t.dtype, dims=t.dims, inits=t[:])
			else:
				raise ValueError(f"InitializeWorkspace can only import scalars and tensors along 'genes' dimension, but '{tensor}' was rank-{t.rank} and first dimension was '{t.dims[0]}'")
