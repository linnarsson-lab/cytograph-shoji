from cytograph.module import requires
from typing import Dict
import shoji
import logging
import sys
import os
import numpy as np
from cytograph import Module
import pandas as pd


class PatchSampleMetadata(Module):
	def __init__(self, table: str, tensors: Dict[str, str], **kwargs) -> None:
		"""
		Args:
			table 				Full path to the metadata database file (a sqlite .db file)
			tensors			Dict of tensors to be loaded from metadata, and their types
		"""
		super().__init__(**kwargs)

		self.table = table
		if not os.path.exists(table):
			logging.error(f"Samples metadata file '{table}' not found")
			sys.exit(1)
		if not table.endswith(".xlsx"):
			logging.error(f"Invalid samples metadata file '{table}' (only .xlsx allowed)")
			sys.exit(1)
		self.tensors = tensors

	@requires("SampleID", "string", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		n_cells = ws.cells.length

		samples = self.SampleID[:]
		for tensor, dtype in self.tensors.items():
			if dtype == "string":
				values = np.full(n_cells, '', dtype="object")
			else:
				values = np.zeros(n_cells, dtype=dtype)
			table = pd.read_excel(self.table)
			for sample in np.unique(samples):
				sampleID = sample.replace("TenX", "10X")
				mask = (samples == sample)
				row = table[table["SampleID"] == sampleID]
				keys = row.columns.values
				try:
					vals = row.values[0]
				except IndexError as e:
					logging.error(e)
					logging.info(f"Tensor {tensor}, SampleID {sample}")
					sys.exit(1)
				d = dict(zip(keys, vals))
				if tensor not in d:
					logging.error(f"'{tensor}' was not found in the metadata")
					sys.exit(1)
				values[mask] = d[tensor]
			logging.info(f" PatchSampleMetadata: Patching metadata for '{tensor}'")
			assert values is not None, f"Failed to load metadata for '{tensor}'"
			ws[tensor] = shoji.Tensor(dtype, ("cells",), inits=values)
