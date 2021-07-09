from cytograph.module import requires
from typing import List, Union, Optional
import shoji
from .config import Config
import logging
import sys
import sqlite3 as sqlite
import os
import numpy as np
from cytograph import Module
import pandas as pd


class PatchSampleMetadata(Module):
	def __init__(self, db: str, tensors: List[str], **kwargs) -> None:
		"""
		Args:
			db 				Full path to the metadata database file (a sqlite .db file)
			tensors			List of tensors to be created (or overwritten) from metadata
		"""
		super().__init__(**kwargs)

		self.db = db
		if not os.path.exists(db):
			logging.error(f"Samples metadata file '{db}' not found")
			sys.exit(1)
		if not db.endswith(".db") and not db.endswith(".xlsx"):
			logging.error(f"Invalid smples metadata file '{db}' (only .db and .xlsx allowed)")
			sys.exit(1)
		self.tensors = tensors

	@requires("SampleID", "string", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		n_cells = ws.cells.length
		def update_values(values: Optional[np.array], mask: np.array, val: Union[int, str, float]):
			if values is None:
				if isinstance(val, int):
					values = np.zeros(n_cells, dtype="int32")
				elif isinstance(val, float):
					values = np.zeros(n_cells, dtype="float32")
				elif isinstance(val, str):
					values = np.zeros(n_cells, dtype="object")
			values[mask] = val
			return values

		samples = self.SampleID[:]
		for tensor in self.tensors:
			values = None
			for sample in np.unique(samples):
				mask = (samples == sample)
				if self.db.endswith(".db"):
					with sqlite.connect(self.db) as sqldb:
						cursor = sqldb.cursor()
						cursor.execute("SELECT * FROM sample WHERE name = ? COLLATE NOCASE", (sample,))
						keys = [x[0].lower() for x in cursor.description]
						vals = cursor.fetchone()
						if vals is None:
							logging.warning(f"Sample '{sample}' not found in database")
						else:
							d = dict(zip(keys, vals))
							if tensor.lower() not in d:
								logging.error(f"Tensor '{tensor}' was not found in the metadata")
								sys.exit(1)
							values = update_values(values, mask, d[tensor.lower()])
				elif self.db.endswith(".xlsx"):
					sampleID = sample.replace("TenX", "10X")
					table = pd.read_excel(self.db)
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
						logging.error(f"Tensor '{tensor}' was not found in the metadata")
						sys.exit(1)
					values = update_values(values, mask, d[tensor])
					values[mask] = d[tensor]
			logging.info(f" PatchSampleMetadata: Patching metadata for '{tensor}'")
			assert values is not None, f"Failed to load metadata for '{tensor}'"
			dtype = values.dtype.name if values.dtype != np.object_ else "string"
			ws[tensor] = shoji.Tensor(dtype, ("cells",), inits=values)
