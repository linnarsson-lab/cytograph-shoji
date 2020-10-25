from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import shoji
from cytograph import requires, creates
from .glmpca_impl import glmpca


class GLMPCA:
	"""
	Project a dataset into a reduced feature space using GLM PCA.
	"""
	def __init__(self, n_factors: int, *, batch_keys: List[str] = None, covariate_keys: List[str] = None) -> None:
		"""
		Args:
			n_factors: 			The number of factors
			batch_keys:			Keys (tensor names) to use as batch keys for batch correction, or None to omit batch correction
			covariate_keys:		Keys (tensor names) to use as numerical covariates (for example, cell cycle scores)
		
		Remarks:
			batch_keys are converted to vectors of 1s and 0s indicating the presence of each unique value
			covariate_keys are left as-is and interpreted as numerical covariates
		"""
		self.n_factors = n_factors
		self.batch_keys = batch_keys
		self.covariate_keys = covariate_keys

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("SelectedFeatures", "bool", ("genes",))
	@creates("GLMPCAFactors", "float32", ("cells", None))
	@creates("GLMPCALoadings", "float32", ("genes", None))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		covariates = None
		if self.batch_keys is not None:
			for key in self.batch_keys:
				x = ws[:][key][:, None]
				onehot = OneHotEncoder(sparse=False).fit_transform(x)
				if covariates is None:
					covariates = onehot
				else:
					covariates = np.hstack([covariates, onehot])

		if self.covariate_keys is not None:
			for key in self.covariate_keys:
				x = ws[:][key][:, None]
				if covariates is None:
					covariates = onehot
				else:
					covariates = np.hstack([covariates, x])

		data = ws[ws.SelectedFeatures == True].Expression
		factors, loadings, _ = glmpca(data.T, self.n_factors, fam="mult", X=covariates, sz=ws[:].TotalUMIs)

		return factors, loadings
