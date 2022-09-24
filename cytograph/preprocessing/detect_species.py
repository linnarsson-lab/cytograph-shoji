from typing import Dict
import shoji
import numpy as np
from cytograph import requires, creates, Algorithm
import logging


class DetectSpecies(Algorithm):
	"""
	Use gene names to detect the species
	"""
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	@requires("Gene", "string", ("genes",))
	@creates("Species", "string", ())
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> str:
		genes = self.Gene[:]
		for gene, species in {
			"ACTB": "Homo sapiens",
			"Tspy1": "Rattus norvegicus",
			"Actb": "Mus musculus",  # Note must come after rat, because rat has the same gene name
			"actb1": "Danio rerio",
			"Act5C": "Drosophila melanogaster",
			"ACT1": "Saccharomyces cerevisiae",
			"act1": "Schizosaccharomyces pombe",
			"act-1": "Caenorhabditis elegans",
			"ACT12": "Arabidopsis thaliana",
			"AFTTAS": "Gallus gallus"
		}.items():
			if gene in genes:
				logging.info(" DetectSpecies: " + species)
				return np.array(species, dtype="object")
		raise ValueError("Could not auto-detect species")

class DetectSpeciesEEL(Algorithm):
	"""
	Use gene names to detect the species
	"""
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	@requires("Gene", "string", ("genes",))
	@creates("Species", "string", ())
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> str:
		genes = self.Gene[:]
		for gene, species in {
			"ACTA2": "Homo sapiens",
			"Tspy1": "Rattus norvegicus",
			"Actb": "Mus musculus",  # Note must come after rat, because rat has the same gene name
			"actb1": "Danio rerio",
			"Act5C": "Drosophila melanogaster",
			"ACT1": "Saccharomyces cerevisiae",
			"act1": "Schizosaccharomyces pombe",
			"act-1": "Caenorhabditis elegans",
			"ACT12": "Arabidopsis thaliana",
			"AFTTAS": "Gallus gallus"
		}.items():
			if gene in genes:
				logging.info(" DetectSpecies: " + species)
				return np.array(species, dtype="object")
		raise ValueError("Could not auto-detect species")
