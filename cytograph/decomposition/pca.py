from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from ..utils import div0
from ..module import creates, requires, Module
import shoji
import logging


class PrincipalComponents(Module):
	"""
	Project a dataset into a reduced feature space using PCA.
	"""
	def __init__(self, n_factors: int = 50, **kwargs) -> None:
		"""
		Args:
			n_factors:  	The number of retained components
		"""
		super().__init__(**kwargs)
		self.n_factors = n_factors

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("SelectedFeatures", "bool", ("genes",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@creates("Factors", "float32", ("cells", None))
	@creates("Loadings", "float32", ("genes", None))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		Expression = self.requires["Expression"]
		SelectedFeatures = self.requires["SelectedFeatures"]
		TotalUMIs = self.requires["TotalUMIs"]

		logging.info(" PrincipalComponents: Loading and normalizing data")
		totals = ws[TotalUMIs][:].astype("float32")
		level = np.median(totals)
		data = ws[ws[SelectedFeatures] == True][Expression]
		vals = np.log2(div0(data.T, totals) * level + 1).T  # Transposed back to (cells, genes)

		logging.info(f" PrincipalComponents: Computing principal components")
		pca = PCA(n_components=self.n_factors)
		factors = pca.fit_transform(vals)
		loadings = pca.components_.T
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, self.n_factors))
		loadings_all[ws.SelectedFeatures[:]] = loadings
		return factors, loadings_all
