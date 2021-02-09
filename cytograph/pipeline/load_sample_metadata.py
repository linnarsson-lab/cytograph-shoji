from typing import List
import shoji
from .config import Config
import logging
import sys
import sqlite3 as sqlite
import os
import numpy as np
from cytograph import Module


# Here's what was in the db on 2020-12-07 for 10X101_1:
#
# Id 411 <class 'int'>
# Name 10X101_1 <class 'str'>
# Shortname XHU:1405:297_Cortex <class 'str'>
# Label  <class 'str'>
# Method rna-seq <class 'str'>
# Project Dev human <class 'str'>
# Donor XHU:1405:297 <class 'str'>
# Species Hs <class 'str'>
# Strain  <class 'str'>
# Age 10.0 <class 'float'>
# Ageunit pcw <class 'str'>
# Agetext 10w <class 'str'>
# Sex  <class 'str'>
# Numpooledanimals  <class 'str'>
# Plugdate  <class 'str'>
# Tissue Cortex <class 'str'>
# Roi  <class 'str'>
# Neuronprop N/A <class 'str'>
# Targetnumcells 5000 <class 'int'>
# Cellconc 1130 <class 'int'>
# Datecaptured 2018-03-29 <class 'str'>
# Sampleok Y <class 'str'>
# Chemistry v2 <class 'str'>
# Transcriptome GRCh38-3.0.0 <class 'str'>
# Comment  <class 'str'>
# Analysis Ready (10X101_1_GRCh38-3_0_0) <class 'str'>
# All_fc_analysis_id 1429 <class 'int'>
# Editby emelie <class 'str'>
# Editat 2019-05-30 16:52:20 <class 'str'>

class LoadSampleMetadata(Module):
	def __init__(self, db: str, tensors: List[str], convert_10x_sample_name: bool = True, **kwargs) -> None:
		"""
		Args:
			db 				Full path to the metadata database file (a sqlite .db file)
			tensors			List of tensors to be created (or overwritten) from metadata
			convert_10x_sample_name		If true, convert "10X101_2" to "TenX101_2"
		"""
		super().__init__(**kwargs)

		self.db = db
		if not os.path.exists(db):
			logging.error(f"Samples metadata file '{db}' not found")
			sys.exit(1)
		self.tensors = tensors
		self.convert_10x_sample_name = convert_10x_sample_name

	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		db = shoji.connect()
		config = Config.load()
		punchcard = config["punchcard"]
		for source in punchcard.sources:
			if source in db[config["workspaces"]["samples"]]:
				source_ws = db[config["workspaces"]["samples"]][source]
			else:
				logging.error(f"Sample {source} not found!")
				sys.exit(1)
			logging.info(f" LoadSampleMetadata: Updating metadata for '{source}'")
			if self.convert_10x_sample_name:
				sample_id = "10X" + source[4:]
			else:
				sample_id = source
			with sqlite.connect(self.db) as sqldb:
				cursor = sqldb.cursor()
				cursor.execute("SELECT * FROM sample WHERE name = ? COLLATE NOCASE", (sample_id,))
				keys = [x[0].lower() for x in cursor.description]
				vals = cursor.fetchone()
				if vals is None:
					logging.warning(f"Sample '{sample_id}' not found in database")
				else:
					d = dict(zip(keys, vals))
					for tensor in self.tensors:
						if tensor.lower() not in d:
							logging.error(f"Tensor '{tensor}' was not found in the metadata")
							sys.exit(1)
						val = d[tensor.lower()]
						if isinstance(val, int):
							source_ws[tensor] = shoji.Tensor("int32", (), np.array(val, dtype="int32"))
						elif isinstance(val, float):
							source_ws[tensor] = shoji.Tensor("float32", (), np.array(val, dtype="float32"))
						elif isinstance(val, str):
							source_ws[tensor] = shoji.Tensor("string", (), np.array(val, dtype=object))
