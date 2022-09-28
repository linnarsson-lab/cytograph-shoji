from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from ..utils import div0
from ..algorithm import creates, requires, Algorithm
import shoji
import logging


class PrincipalComponents(Algorithm):
	"""
	Project a dataset into a reduced feature space using PCA.
	"""
	def __init__(self, n_factors: int = 50, **kwargs) -> None:
		"""
		Args:
			n_factors:  	The number of retained components
		
		Remarks:
			This algorithm loads all the data for selected features, which may require 
			a large amount of memory
		"""
		super().__init__(**kwargs)
		self.n_factors = n_factors

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@creates("Factors", "float32", ("cells", None))
	@creates("Loadings", "float32", ("genes", None))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		logging.info(" PrincipalComponents: Loading and normalizing data")
		totals = self.TotalUMIs[:].astype("float32")
		level = np.median(totals)
		data = self.Expression[:, self.SelectedFeatures == True]
		vals = np.log2(div0(data.T, totals) * level + 1).T  # Transposed back to (cells, genes)
		#vals = scale(vals)

		logging.info(f" PrincipalComponents: Computing principal components")
		pca = PCA(n_components=self.n_factors)
		factors = pca.fit_transform(vals)

		evs = ", ".join([f"{x:.2f}" for x in pca.explained_variance_ratio_ if x > 0.01]) + ", ..."
		logging.info(f" PrincipalComponents: Explained variance ({int(pca.explained_variance_ratio_.sum() * 100)}%): {evs}")

		keep_factors = 50
		if pca.explained_variance_ratio_.sum() > 0.75:
			keep_factors = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.5)[0])

		logging.info(f" PrincipalComponents: Keeping {keep_factors} components that explain {int(pca.explained_variance_ratio_[:keep_factors].sum() * 100)}% of variance")
		loadings = pca.components_.T
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, keep_factors))
		loadings_all[ws.SelectedFeatures[:]] = loadings[:, :keep_factors]
		return factors[:, :keep_factors], loadings_all
