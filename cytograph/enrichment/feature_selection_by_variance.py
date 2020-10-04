import numpy as np
from sklearn.svm import SVR
from cytograph import CytographMethod
import shoji
import logging


class FeatureSelectionByVariance(CytographMethod):
	def __init__(self, n_genes: int, mask: np.ndarray = None) -> None:
		"""
		Args:
			n_genes		Number of genes to select
			mask		Optional mask (numpy bool array) indicating genes that should not be selected
		"""
		self.n_genes = n_genes
		self.mask = mask
		self._requires = [
			("Expression", None, ("cells", "genes")),
			("MeanExpression", "float32", ("genes",)),
			("StdevExpression", "float32", ("genes",))
		]

	def fit(self, ws: shoji.WorkspaceManager) -> np.ndarray:
		"""
		Fits a noise model (CV vs mean) and select high-variance genes

		Args:
			ws:	shoji.Workspace containing the data to be used

		Returns:
			ndarray of selected genes (bool array)
		
		Remarks:
			If the tensor "ValidGenes" exists, only ValidGenes == True genes will be selected
		"""
		self.check(ws, "FeatureSelectionByVariance")
		n_genes = ws.genes.length

		logging.info("FeatureSelectionByVariance: Fitting CV vs mean")
		mu = ws[:].MeanExpression
		sd = ws[:].StdevExpression

		if "ValidGenes" in ws:
			valid = ws.ValidGenes[:]
		else:
			valid = np.ones(n_genes, dtype='bool')
		if self.mask is not None:
			valid = np.logical_and(valid, np.logical_not(self.mask))

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
		selected = np.zeros(n_genes, dtype=bool)
		selected[np.sort(genes)] = True
		logging.info("FeatureSelectionByVariance: Done.")
		return selected

	def fit_save(self, ws: shoji.WorkspaceManager) -> np.ndarray:
		"""
		Fits a noise model (CV vs mean) and select high-variance genes, save the
		result as bool tensor SelectedFeatures.

		Returns:
			Bool array indicating the selected genes
		"""
		selected = self.fit(ws)
		logging.info("FeatureSelectionByVariance: Saving selected features as bool tensor 'SelectedFeatures'")
		ws.SelectedFeatures = shoji.Tensor("bool", ("genes",), selected)
		return selected
