from typing import List, Dict
import shoji
import numpy as np
from .config import Config
import logging
import cytograph as cg
import sys


class CollectCells:
	def __init__(self, tensors: List[str], expand_scalars: bool = True) -> None:
		"""
		Args:
			tensors			List of tensors to be collected (must exist and have same dims and dtype in all samples)
			expand_scalars	If true, scalars are converted to vectors (repeating the scalar value)
		"""
		self.tensors = tensors
		self.expand_scalars = expand_scalars

	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		"""
		Collect cells from a list of workspaces (defined in config.include) into this workspace

		Args:
			ws				shoji workspace
		"""
		db = shoji.connect()
		config = Config.load()
		punchcard = config["punchcard"]
		for ix, source in enumerate(punchcard.sources):
			if source in config["workspaces"]["build"]:
				source_ws = db[config["workspace"]][source]
			elif source in db[config["workspaces"]["samples"]]:
				source_ws = db[config["workspaces"]["samples"]][source]
			d: Dict[str, np.ndarray] = {}
			onlyif = punchcard.sources_onlyif[ix]
			if onlyif is not None:
				logging.info(f" CollectCells: Collecting tensors from '{source}' where '{onlyif}'")
				conditions = eval(onlyif, {"ws": source_ws, "np": np, "shoji": shoji, "cg": cg})
				if not isinstance(conditions, shoji.Filter):
					raise ValueError(f"Conditions in 'onlyif' must evaluate to a shoji.Filter, but '{onlyif}' evaluated to '{type(conditions)}'")
				view = source_ws[conditions, ...]
			else:
				logging.info(f" CollectCells: Collecting tensors from '{source}'")
				view = source_ws[...]
			for tensor in self.tensors:
				if tensor not in source_ws:
					logging.error(f"Tensor '{tensor}' missing in source workspace '{source}")
					sys.exit(1)
				t = source_ws[tensor]
				if t.rank > 0 and t.dims[0] == "cells":
					if ix == 0:
						ws[tensor] = shoji.Tensor(t.dtype, t.dims)
					d[tensor] = view[tensor]
				elif t.rank == 0:
					if self.expand_scalars:
						if ix == 0:
							ws[tensor] = shoji.Tensor(t.dtype, ("cells",))
						d[tensor] = np.full(view.get_length("cells"), t[:], dtype=t.numpy_dtype())
					else:
						ws[tensor] = shoji.Tensor(t.dtype, t.dims, t[:])
			ws.cells.append(d)
		logging.info(f" CollectCells: Collected {ws.cells.length} cells")
		ws.cells = shoji.Dimension(shape=ws.cells.length)  # Fix the length of the cells dimension
