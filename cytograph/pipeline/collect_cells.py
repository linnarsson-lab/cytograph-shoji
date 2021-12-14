from cytograph.pipeline.punchcards import Punchcard
from typing import List
import shoji
import numpy as np
from .config import Config
import logging
import cytograph as cg
from cytograph import Module
import sys
from sklearn.preprocessing import LabelEncoder


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
		Collect cells from a list of workspaces (defined in config.sources) into this workspace

		Args:
			ws				shoji workspace
		"""
		db = shoji.connect()
		config = Config.load()
		punchcard = config.punchcard
		assert punchcard is not None
		build_ws = config.workspaces.build
		assert build_ws is not None

		for ix, source in enumerate(punchcard.sources):
			assert build_ws is not None
			if source != punchcard.name and source in build_ws:
				source_ws = build_ws[source]
			elif source in db[config.workspaces.samples_workspace_name]:
				source_ws = db[config.workspaces.samples_workspace_name][source]
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
				indices = view.filters["cells"].get_rows(source_ws)
			else:
				logging.info(f" CollectCells: Collecting tensors from '{source}'")
				view = source_ws[:]
				indices = np.arange(source_ws.cells.length)
			if punchcard.with_annotation is not None:
				logging.info(f" CollectCells: Keeping only cells that have auto-annotation '{punchcard.with_annotation}'")
				if "AnnotationPosterior" not in source_ws:
					raise ValueError(f"Punchcard uses 'with_annotation' but source '{source}' lacks auto-annotation")
				pp = source_ws.AnnotationPosterior[:, source_ws.AnnotationName == punchcard.with_annotation]
				keep_clusters = source_ws.ClusterID[pp > 0.95]
				labels = source_ws.Clusters[:]
				aa_indices = np.array([], dtype="uint32")
				for cluster in keep_clusters:
					aa_indices = np.union1d(aa_indices, np.where(labels == cluster)[0])
				indices = np.intersect1d(indices, aa_indices)
				logging.info(f" CollectCells: Keeping {indices.shape[0]} cells from {keep_clusters.shape[0]} clusters with '{punchcard.with_annotation}'")
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
					if new_name in self.renumber_tensors and t.rank != 1:
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
			offset = 0  # Offset of cluster IDs
			ix = 0  # Index of cells
			for source in punchcard.sources:
				logging.info(f" CollectCells: Renumbering '{tensor}' from '{source}' with offset={offset}")
				if source != punchcard.name and source in build_ws:
					source_ws = build_ws[source]
				elif source in db[config.workspaces.samples_workspace_name]:
					source_ws = db[config.workspaces.samples_workspace_name][source]
				if onlyif is not None:
					conditions = eval(onlyif, {"ws": source_ws, "np": np, "shoji": shoji, "cg": cg})
					view = source_ws[conditions]
				else:
					view = source_ws[:]
				vals = LabelEncoder().fit_transform(view[tensor])
				ws[tensor][ix:ix + vals.shape[0]] = vals + offset
				offset = offset + max(vals) + 1
				ix += vals.shape[0]

		ws.cells = shoji.Dimension(shape=ws.cells.length)  # Fix the length of the cells dimension
		logging.info(f" CollectCells: Collected {ws.cells.length} cells")
