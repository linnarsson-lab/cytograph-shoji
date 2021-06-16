from typing import List
import shoji
import numpy as np
from .config import Config
import logging
import cytograph as cg
from cytograph import Module
import sys


class CollectCells(Module):
	def __init__(self, tensors: List[str], expand_scalars: bool = True, renumber_tensors: List[str] = None, **kwargs) -> None:
		"""
		Args:
			tensors				List of tensors to be collected (must exist and have same dims and dtype in all samples)
			expand_scalars		If true, scalars are converted to vectors (repeating the scalar value)
			renumber_tensors	List of tensors that should be renumbered to stay unique while combining sources (e.g. "Clusters")

		Remarks:
			Tensors can be renamed on the fly using the A->B syntax, e.g. "SampleName->SampleID"
		"""
		super().__init__(**kwargs)

		self.tensors = tensors
		self.expand_scalars = expand_scalars
		self.renumber_tensors = renumber_tensors if renumber_tensors is not None else []

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
			if source != punchcard.name and source in config["workspaces"]["build"]:
				source_ws = config["workspaces"]["build"][source]
			elif source in db[config["workspaces"]["samples"]]:
				source_ws = db[config["workspaces"]["samples"]][source]
			else:
				logging.error(f"Source {source} not found!")
				sys.exit(1)
			onlyif = punchcard.sources_onlyif[ix]
			if onlyif is not None:
				logging.info(f" CollectCells: Collecting tensors from '{source}' where '{onlyif}'")
				conditions = eval(onlyif, {"ws": source_ws, "np": np, "shoji": shoji, "cg": cg})
				if not isinstance(conditions, shoji.Filter):
					raise ValueError(f"Conditions in 'onlyif' must evaluate to a shoji.Filter, but '{onlyif}' evaluated to '{type(conditions)}'")
				view = source_ws[conditions]
			else:
				logging.info(f" CollectCells: Collecting tensors from '{source}'")
				view = source_ws[:]

			indices = view.filters["cells"].get_rows(source_ws)
			batch_size = 5_000
			for start in range(0, indices.shape[0], batch_size):
				d = {}
				for tensor_spec in self.tensors:
					if "->" in tensor_spec:
						tensor, new_name = tensor_spec.split("->")
					else:
						tensor, new_name = tensor_spec, tensor_spec
					if tensor not in source_ws:
						logging.error(f"Tensor '{tensor}' missing in source workspace '{source}")
						sys.exit(1)
					t = source_ws[tensor]
					if tensor in self.renumber_tensors and t.rank != 1:
						logging.error(f"Cannot renumber tensor '{tensor}' because rank is not 1")
						sys.exit(1)
					if t.rank > 0:
						if t.dims[0] != "cells":
							logging.error(f"Cannot collect tensor '{tensor}' because first dimension is not 'cells'")
							sys.exit(1)
						if ix == 0 and start == 0:
							ws[new_name] = shoji.Tensor(t.dtype, t.dims)
						d[new_name] = source_ws[tensor][indices[start: start + batch_size]]
					elif t.rank == 0:
						if self.expand_scalars:
							if ix == 0 and start == 0:
								ws[new_name] = shoji.Tensor(t.dtype, ("cells",))
							d[new_name] = np.full(min(indices.shape[0] - start, batch_size), t[:], dtype=t.numpy_dtype())
						elif ix == 0 and start == 0:
							ws[new_name] = shoji.Tensor(t.dtype, t.dims, inits=t[:])
				ws.cells.append(d)
				start += batch_size

		for tensor in self.renumber_tensors:
			logging.info(f" CollectCells: Renumbering '{source}'")
			offset = 0
			for source in punchcard.sources:
				t = db[config["workspaces"]["samples"]][source][tensor]
				vals = t[:]
				t[:] = vals + offset
				offset = offset + max(vals) + 1

		ws.cells = shoji.Dimension(shape=ws.cells.length)  # Fix the length of the cells dimension
		logging.info(f" CollectCells: Collected {ws.cells.length} cells")
