from cytograph.module import requires
from typing import Dict
import shoji
import logging
import sys
import os
import numpy as np
from cytograph import Module
import pandas as pd


class PatchGeneMetadata(Module):
	def __init__(self, table: str, patch_accession_from: str = None, **kwargs) -> None:
		"""
		Args:
			table 			Full path to the metadata database file (a sqlite .db file)
		"""
		super().__init__(**kwargs)

		self.table = table
		self.patch_accession_from = patch_accession_from
		if not os.path.exists(table):
			logging.error(f"Gene metadata file '{table}' not found")
			sys.exit(1)

	@requires("Accession", "string", ("genes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		metadata = pd.read_csv(self.table, delimiter="\t", keep_default_na=False)

		if self.patch_accession_from is not None:
			logging.info(f" PatchGeneMetadata: Patching accessions from {self.patch_accession_from}")
			db = shoji.connect()
			ws.Accession = shoji.Tensor("string", ("genes",), inits=db.samples_ih[self.patch_accession_from].Accession[:])

		logging.info(f" PatchGeneMetadata: Loading gene metadata")
		accessions = pd.DataFrame({"Accession": ws.Accession[:]})
		accessions = accessions.set_index("Accession")
		metadata = metadata.drop_duplicates("AccessionVersion")
		metadata.set_index("AccessionVersion")
		joined = pd.merge(accessions, metadata, left_on="Accession", right_on="AccessionVersion", how="left", validate="many_to_one")

		logging.info(f" PatchGeneMetadata: Patching gene metadata")
		ws["GeneFullName"] = shoji.Tensor("string", ("genes",), inits=joined["FullName"].fillna("").values)
		ws["GeneType"] = shoji.Tensor("string", ("genes",), inits=joined["GeneType"].fillna("").values)
		ws["LocusGroup"] = shoji.Tensor("string", ("genes",), inits=joined["LocusGroup"].fillna("").values)
		ws["LocusType"] = shoji.Tensor("string", ("genes",), inits=joined["LocusType"].fillna("").values)
		ws["Location"] = shoji.Tensor("string", ("genes",), inits=joined["Location"].fillna("").values)
		ws["LocationSortable"] = shoji.Tensor("string", ("genes",), inits=joined["LocationSortable"].fillna("").values)
		ws["Aliases"] = shoji.Tensor("string", ("genes",), inits=joined["Aliases"].fillna("").values)
		ws["IsTF"] = shoji.Tensor("bool", ("genes",), inits=joined["IsTF"].fillna(False).values.astype(bool))
		ws["DnaBindingDomain"] = shoji.Tensor("string", ("genes",), inits=joined["DnaBindingDomain"].fillna("").values)
		ws["Chromosome"] = shoji.Tensor("string", ("genes",), inits=joined["Chromosome"].fillna("").values)
		ws["ChromosomeStart"] = shoji.Tensor("string", ("genes",), inits=joined["ChromosomeStart"].fillna("").values)
		ws["ChromosomeEnd"] = shoji.Tensor("string", ("genes",), inits=joined["ChromosomeEnd"].fillna("").values)
