import logging
from typing import Callable, Union
from cytograph import requires, creates, Algorithm
import numpy as np
import shoji
import umap
from ..utils import available_cpu_count


class UMAP(Algorithm):
	"""
	Uniform Manifold Approximation and Projection for Dimension Reduction
	"""
	def __init__(
			self, 
			metric: Union[str, Callable] = "euclidean", 
			n_neighbors: int = 15, 
			min_dist: float = 0.1, 
			density_regularization: float = 0, 
			**kwargs):
		"""
		Uniform Manifold Approximation and Projection for Dimension Reduction.
		See https://umap-learn.readthedocs.io/en/latest/index.html

		Args:
			metric						Any metric allowed by Annoy (default: 'euclidean')
			density_regularization		Use DensMAP (https://www.biorxiv.org/content/10.1101/2020.05.12.077776v1) to regularize local density (default: 0)
		"""
		super().__init__(**kwargs)
		self.metric = metric
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.density_regularization = density_regularization

	@requires("Factors", "float32", ("cells", None))
	@creates("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		return self._fit(self.Factors[:])

	def _fit(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
		"""
		Returns:
			The UMAP embedding as np.ndarray
		"""
		logging.info(" UMAP: Computing the embedding")
		Z = umap.UMAP(
			metric=self.metric,
			n_neighbors=self.n_neighbors,
			min_dist=self.min_dist,
			dens_lambda=self.density_regularization,
			densmap=self.density_regularization > 0,
			n_jobs=available_cpu_count()
		).fit_transform(X, y=y)
		return Z

class SupervisedUMAP(Algorithm):
	"""
	Uniform Manifold Approximation and Projection for Dimension Reduction
	"""
	def __init__(
			self, 
			metric: Union[str, Callable] = "euclidean", 
			n_neighbors: int = 15, 
			min_dist: float = 0.1, 
			density_regularization: float = 0, 
			**kwargs):
		"""
		Uniform Manifold Approximation and Projection for Dimension Reduction.
		See https://umap-learn.readthedocs.io/en/latest/index.html

		Args:
			metric						Any metric allowed by Annoy (default: 'euclidean')
			density_regularization		Use DensMAP (https://www.biorxiv.org/content/10.1101/2020.05.12.077776v1) to regularize local density (default: 0)
		"""
		super().__init__(**kwargs)
		self.metric = metric
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.density_regularization = density_regularization

	@requires("Factors", "float32", ("cells", None))
	@requires("GraphCluster", "int8", ("cells", ))
	@creates("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		labels = self.GraphCluster[:]
		for gi in np.unique(labels):
			if (labels == gi).sum() > 50:
				pass
			else:
				labels[labels == gi] = -1
		return self._fit(self.Factors[:], labels)

	def _fit(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
		"""
		Returns:
			The UMAP embedding as np.ndarray
		"""
		logging.info(" UMAP: Computing the embedding")
		Z = umap.UMAP(
			metric=self.metric,
			n_neighbors=self.n_neighbors,
			min_dist=self.min_dist,
			dens_lambda=self.density_regularization,
			densmap=self.density_regularization > 0,
			n_jobs=available_cpu_count()
		).fit_transform(X, y=y)
		return Z