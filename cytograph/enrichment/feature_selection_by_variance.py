from typing import List
import numpy as np
from sklearn.svm import SVR
import cytograph as cg
from cytograph import requires, creates
import shoji
import logging


class FeatureSelectionByVariance:
	def __init__(self, n_genes: int, mask: List[str] = None) -> None:
		"""
		Args:
			n_genes		Number of genes to select
			mask		Optional mask (numpy bool array) indicating genes that should not be selected
		"""
		self.n_genes = n_genes
		self.mask = mask if mask is not None else []

	@requires("Species", "string", ())
	@requires("Expression", None, ("cells", "genes"))
	@requires("MeanExpression", "float32", ("genes",))
	@requires("StdevExpression", "float32", ("genes",))
	@creates("SelectedFeatures", "bool", ("genes",), indices=True)
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		"""
		Fits a noise model (CV vs mean) and select high-variance genes

		Args:
			ws:	shoji.Workspace containing the data to be used

		Returns:
			ndarray of indices of selected genes
		
		Remarks:
			If the tensor "ValidGenes" exists, only ValidGenes == True genes will be selected
		"""
		n_genes = ws.genes.length
		species = cg.Species(ws[:].Species)
		mask_genes = species.mask(ws, self.mask)

		logging.info("FeatureSelectionByVariance: Fitting CV vs mean")
		mu = ws[:].MeanExpression
		sd = ws[:].StdevExpression

		if "ValidGenes" in ws:
			valid = ws.ValidGenes[:]
		else:
			valid = np.ones(n_genes, dtype='bool')
		if self.mask is not None:
			valid = np.logical_and(valid, np.logical_not(mask_genes))

		ok = np.logical_and(mu > 0, sd > 0)
		cv = sd[ok] / mu[ok]
		log2_m = np.log2(mu[ok])
		log2_cv = np.log2(cv)

		svr_gamma = 1000. / len(mu[ok])
		clf = SVR(gamma=svr_gamma)
		clf.fit(log2_m[:, np.newaxis], log2_cv)
		fitted_fun = clf.predict
		# Score is the relative position with respect of the fitted curve
		score = log2_cv - fitted_fun(log2_m[:, np.newaxis])
		score = score * valid[ok]
		genes = np.where(ok)[0][np.argsort(score)][-self.n_genes:]
		logging.info("FeatureSelectionByVariance: Done.")
		return genes
