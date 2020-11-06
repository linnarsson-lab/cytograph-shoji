from typing import Tuple
import shoji
import numpy as np
import cytograph as cg
from cytograph import requires, creates
import logging


class DetectSpecies:
	def __init__(self) -> None:
		pass

	@requires("Gene", "string", ("genes",))
	@creates("Species", "string", ())
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> str:
		genes = ws[:].Gene
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
