import numpy as np
import tomotopy as tt
from ..utils import available_cpu_count
import scipy.sparse as sparse
import logging
import shoji
from typing import Union, List, Tuple, Optional
from ..algorithm import creates, requires, Algorithm
from sklearn.decomposition import LatentDirichletAllocation

# Wrapper around the tomotopy LDAModel, with sane API
class LDA(Algorithm):
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
		
		X = self.Expression[:, self.SelectedFeatures == True]
		lda = LatentDirichletAllocation(n_components=self.n_factors, max_iter=10, learning_method='online', learning_offset=50.,random_state=0).fit(X)
		factors = lda.transform(X)
		loadings = lda.components_
		loadings_all = np.zeros_like(loadings, shape=(ws.genes.length, self.n_factors))
		loadings_all[ws.SelectedFeatures[:]] = loadings.T#[:, :self.n_factors]
		logging.info('Loadings shape: {}'.format(loadings_all.shape))
		return factors, loadings_all

