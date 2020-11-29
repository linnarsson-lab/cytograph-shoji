from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder

import shoji
import numpy as np
from cytograph import requires, creates
from .glmpca_impl import glmpca
import sys
import logging


class GLMPCA:
	"""
	Project a dataset into a reduced feature space using GLM PCA.
	"""
	def __init__(self, n_factors: int, *, family: str = "poi", epsilon: float = 1e-4, penalty: float = 1.0, batch_keys: List[str] = None, covariate_keys: List[str] = None) -> None:
		"""
		This class implements the GLM-PCA dimensionality reduction method for high-dimensional count data.

		Args:
			n_factors: 			The number of factors
			family:				The shape of the likelihood ("poi" for Poisson, "nb" for Negative Binomial, "mult" for binomial approximation to multinomial, "bern" for Bernoulli)
			epsilon: 			Maximum relative change in deviance allowed at convergence
			penalty:			The L2 penalty for the latent factors (covariates are not penalized)
			batch_keys:			Keys (tensor names) to use as batch keys for batch correction, or None to omit batch correction
			covariate_keys:		Keys (tensor names) to use as numerical covariates (for example, cell cycle scores)

		Remarks:
			The basic model is R = AX'+ZG'+VU', where E[Y]=M=linkinv(R). Regression coefficients are A and G,
			latent factors are U, and loadings are V. The objective function being optimized is the deviance between
			Y and M, plus an L2 (ridge) penalty on U and V. Note that glmpca uses a random initialization, so
			for fully reproducible results one should set the random seed.

			batch_keys are converted to vectors of 1s and 0s indicating the presence of each unique value
			covariate_keys are left as-is and interpreted as numerical covariates
		"""
		self.n_factors = n_factors
		if not isinstance(batch_keys, (tuple, list)):
			logging.error(" GLMPCA: 'batch_keys' must be a list of tensor names")
			sys.exit(1)
		self.batch_keys = batch_keys
		if not isinstance(batch_keys, (tuple, list)):
			logging.error(" GLMPCA: 'covariate_keys' must be a list of tensor names")
			sys.exit(1)
		self.covariate_keys = covariate_keys
		self.family = family
		self.epsilon = float(epsilon)  # We cast to float in case the user specified epsilon as a string (since yaml doesn't support 1e-4 notation)
		self.penalty = penalty

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@creates("GLMPCAFactors", "float32", ("cells", None))
	@creates("GLMPCALoadings", "float32", ("genes", None))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		covariates = None
		if self.batch_keys is not None:
			for key in self.batch_keys:
				x = ws[key][...][:, None]
				onehot = OneHotEncoder(sparse=False).fit_transform(x)[:, :-1]  # Drop the last column since it's redundant and will make the GLM non-identifiable
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
		if covariates is not None and covariates.shape[1] == 0:
			covariates = None
		size_factors = ws.TotalUMIs[...].astype("float32")
		size_factors = size_factors / np.median(size_factors)
		data = ws[ws.SelectedFeatures == True, ...].Expression
		if covariates is None:
			logging.info(f" GLMPCA: Optimizing {data.shape[0]} cells × {data.shape[1]} genes with zero covariates")
		elif covariates.shape[1] == 1:
			logging.info(f" GLMPCA: Optimizing {data.shape[0]} cells × {data.shape[1]} genes with one covariate")
		else:
			logging.info(f" GLMPCA: Optimizing {data.shape[0]} cells × {data.shape[1]} genes with {covariates.shape[1]} covariates")
		factors, loadings, _ = glmpca(data.T, self.n_factors, ctl = {"maxIter": 1000, "eps": self.epsilon, "optimizeTheta": True}, penalty=self.penalty, fam=self.family, X=covariates, sz=size_factors)
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, self.n_factors))
		loadings_all[ws.SelectedFeatures[...]] = loadings
		return factors, loadings_all
