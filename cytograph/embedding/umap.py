import logging
from typing import Callable, Union
from cytograph import requires, creates, Module
import numpy as np
import shoji
import umap


class UMAP(Module):
	def __init__(self, n_components: int = 2, metric: Union[str, Callable] = "euclidean", n_neighbors: int = 15, min_dist: float = 0.1, density_regularization: float = 0, **kwargs):
		"""
		Uniform Manifold Approximation and Projection for Dimension Reduction.
		See https://umap-learn.readthedocs.io/en/latest/index.html

		Args:
			metric						Any metric allowed by Annoy (default: 'euclidean')
			density_regularization		Use DensMAP (https://www.biorxiv.org/content/10.1101/2020.05.12.077776v1) to regularize local density (default: 0)
		"""
		super().__init__(**kwargs)
		self.metric = metric
		self.n_components = n_components
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.density_regularization = density_regularization

	@requires("Factors", "float32", ("cells", None))
	@creates("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		return self._fit(self.Factors[:])

	def _fit(self, X: np.ndarray) -> np.ndarray:
		"""
		Returns:
			The UMAP embedding as np.ndarray
		"""
		logging.info(f" UMAP: Computing the embedding")
		exaggeration = self.exaggeration
		if exaggeration == -1:
			exaggeration = 1
		# Just a plain TSNE with high learning rate
		lr = max(200, n / 12)
		aff = affinity.PerplexityBasedNN(
			X,
			perplexity=self.perplexity,
			metric=self.metric,
			method="annoy",
			n_jobs=-1
		)

		init = init_method(X)

		Z = TSNEEmbedding(
			init,
			aff,
			learning_rate=lr,
			n_jobs=-1,
			negative_gradient_method="fft"
		)
		Z.optimize(250, exaggeration=12, momentum=0.5, inplace=True, n_jobs=-1)
		Z.optimize(750, exaggeration=exaggeration, momentum=0.8, inplace=True, n_jobs=-1)
		return Z
