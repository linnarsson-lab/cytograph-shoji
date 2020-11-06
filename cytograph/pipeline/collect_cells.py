from typing import List, Dict
import shoji
import numpy as np
from .config import load_config
import logging


class CollectCells:
	def __init__(self, tensors: List[str]) -> None:
		"""
		Args:
			tensors		List of tensors to be collected (must exist and have same dims and dtype in all samples)
		"""
		config = load_config()
		self.sources = config["include"]
		self.tensors = tensors
		self.conditions = config["onlyif"]

	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		"""
		Collect cells from a list of workspaces (defined in config.include) into this workspace

		Args:
			ws				shoji workspace
		"""
		db = shoji.connect()
		config = load_config()
		for source in self.sources:
			if source in db[config["paths"]["samples"]]:
				source_ws = db[config["paths"]["samples"]][source]
			elif source in db[config["paths"]["workspace"]]:
				source_ws = db[config["paths"]["workspace"]][source]
			d: Dict[str, np.ndarray] = {}
			if self.conditions != "":
				logging.info(f" CollectCells: Collecting tensors from '{source}' where '{self.conditions}'")
				names = {}
				for t in source_ws._tensors():
					names[t] = source_ws[t]
				conditions = eval(self.conditions, names)
				if not isinstance(conditions, shoji.Filter):
					raise ValueError(f"Conditions in 'onlyif' must evaluate to a shoji.Filter, but '{self.conditions}' evaluated to '{type(conditions)}'")
				view = source_ws[conditions]
			else:
				logging.info(f" CollectCells: Collecting tensors from '{source}'")
				view = source_ws[:]
			for tensor in self.tensors:
				t = source_ws[tensor]
				if t.rank > 0 and t.dims[0] == "cells":
					d[tensor] = view[tensor]
				elif t.rank == 0:
					ws[tensor] = t[:]
			ws.cells.append(d)
