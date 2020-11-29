from typing import List
import numpy as np
from sklearn.svm import SVR
import cytograph as cg
from cytograph import requires, creates
import shoji
import logging


class FeatureSelectionByDeviance:
	def __init__(self, n_genes: int, mask: List[str] = None) -> None:
		"""
		Args:
			n_genes		Number of genes to select
			mask		Optional list indicating categories of genes that should not be selected
		"""
		self.n_genes = n_genes
		self.mask = mask if mask is not None else []

	@requires("Species", "string", ())
	@requires("Expression", None, ("cells", "genes"))
	@requires("Deviance", "float32", ("genes",))
	@creates("SelectedFeatures", "bool", ("genes",), indices=True)
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		"""
		Selects genes that show high deviance using a binomial model

		Args:
			ws:	shoji.Workspace containing the data to be used

		Returns:
			ndarray of indices of selected genes
		
		Remarks:
			If the tensor "ValidGenes" exists, only ValidGenes == True genes will be selected
			See equation D_j on p. 14 of https://doi.org/10.1186/s13059-019-1861-6
		"""
		n_genes = ws.genes.length
		species = cg.Species(ws[:].Species)
		mask_genes = species.mask(ws, self.mask)

		logging.info("FeatureSelectionByDeviance: Loading deviance")
		d = ws[:].Deviance

		logging.info("FeatureSelectionByDeviance: Removing invalid and masked genes")
		if "ValidGenes" in ws:
			valid = ws.ValidGenes[:]
		else:
			valid = np.ones(n_genes, dtype='bool')
		if self.mask is not None:
			valid = np.logical_and(valid, np.logical_not(mask_genes))

		temp = []
		for gene in np.argsort(-d):
			if valid[gene]:
				temp.append(gene)
			if len(temp) >= self.n_genes:
				break
		genes = np.array(temp)
		logging.info(f"FeatureSelectionByDeviance: Selected the top {len(genes)} genes")
		return genes
