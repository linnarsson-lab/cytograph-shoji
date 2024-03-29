import logging
from typing import Callable, Union
from cytograph import requires, creates, Algorithm
import numpy as np
from openTSNE import TSNEEmbedding, affinity, initialization
import shoji
from ..utils import available_cpu_count


class ArtOfTsne(Algorithm):
	"""
	t-Stochastic Neighborhood Embedding (TSNE)
	"""
	def __init__(self, metric: Union[str, Callable] = "euclidean", exaggeration: float = -1, perplexity: int = 30, init_method: Union[str, Callable] = "pca", **kwargs):
		"""
		Implementation of Dmitry Kobak and Philipp Berens "The art of using t-SNE for single-cell transcriptomics" based on openTSNE.
		See https://doi.org/10.1038/s41467-019-13056-x | www.nature.com/naturecommunications

		Args:
			metric			Any metric allowed by Annoy (default: 'euclidean')
			exaggeration	The exaggeration to use for the embedding or -1 to use automatic heuristic (default: -1)
			perplexity		The perplexity to use for the embedding
			init_method		Either 'pca', 'random', or a custom method with the same signature as initialization.pca.
		"""
		super().__init__(**kwargs)
		self.metric = metric
		self.exaggeration = exaggeration
		self.perplexity = perplexity
		self.init_method = init_method

	@requires("Factors", "float32", ("cells", None))
	@creates("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		return self._fit(self.Factors[:])

	def _fit(self, X: np.ndarray) -> np.ndarray:
		"""
		Returns:
			The 2D tSNE embedding as np.ndarray
		"""
		if self.init_method == "random":
			init_method = initialization.random
		elif self.init_method == "pca":
			init_method = initialization.pca
		n = X.shape[0]
		if n > 100_000:
			exaggeration = self.exaggeration
			if self.exaggeration == -1:
				exaggeration = 1 + n / 333_333
			# Subsample, optimize, then add the remaining cells and optimize again
			# Also, use exaggeration == 4
			logging.info(f" ArtOfTsne: Using large-scale heuristics (n > 100,000)")
			logging.info(f" ArtOfTsne: Creating subset of {n // 40:,} elements")
			# Subsample and run a regular art_of_tsne on the subset
			indices = np.random.permutation(n)
			reverse = np.argsort(indices)
			X_sample, X_rest = X[indices[:n // 40]], X[indices[n // 40:]]
			logging.info(f" ArtOfTsne: Embedding subset")
			Z_sample = self._fit(X_sample)

			logging.info(f" ArtOfTsne: Preparing partial initial embedding of the {n - n // 40:,} remaining elements")
			if isinstance(Z_sample.affinities, affinity.Multiscale):
				rest_init = Z_sample.prepare_partial(X_rest, k=1, perplexities=[1 / 3, 1 / 3])
			else:
				rest_init = Z_sample.prepare_partial(X_rest, k=1, perplexity=1 / 3)
			logging.info(f" ArtOfTsne: Combining the initial embeddings, and standardizing")
			init_full = np.vstack((Z_sample, rest_init))[reverse]
			init_full = init_full / (np.std(init_full[:, 0]) * 10000)

			logging.info(f" ArtOfTsne: Creating multiscale affinities")
			affinities = affinity.PerplexityBasedNN(
				X,
				perplexity=self.perplexity,
				metric=self.metric,
				method="annoy",
				n_jobs=available_cpu_count()
			)

			logging.info(f" ArtOfTsne: Creating TSNE embedding")
			Z = TSNEEmbedding(
				init_full,
				affinities,
				negative_gradient_method="fft",
				n_jobs=available_cpu_count()
			)
			logging.info(f" ArtOfTsne: Optimizing, stage 1")
			Z.optimize(n_iter=250, inplace=True, exaggeration=12, momentum=0.5, learning_rate=n / 12, n_jobs=-1)
			logging.info(f" ArtOfTsne: Optimizing, stage 2")
			Z.optimize(n_iter=750, inplace=True, exaggeration=exaggeration, momentum=0.8, learning_rate=n / 12, n_jobs=-1)
		elif n > 3_000:
			logging.info(f" ArtOfTsne: Using medium-scale heuristics (3,000 < n < 100,000)")
			exaggeration = self.exaggeration
			if exaggeration == -1:
				exaggeration = 1
			# Use multiscale perplexity
			affinities_multiscale_mixture = affinity.Multiscale(
				X,
				perplexities=[self.perplexity, n / 100],
				metric=self.metric,
				method="annoy",
				n_jobs=available_cpu_count()
			)
			init = init_method(X)
			Z = TSNEEmbedding(
				init,
				affinities_multiscale_mixture,
				negative_gradient_method="fft",
				n_jobs=available_cpu_count()
			)
			Z.optimize(n_iter=250, inplace=True, exaggeration=12, momentum=0.5, learning_rate=n / 12, n_jobs=-1)
			Z.optimize(n_iter=750, inplace=True, exaggeration=exaggeration, momentum=0.8, learning_rate=n / 12, n_jobs=-1)
		else:
			logging.info(f" ArtOfTsne: Using small-scale heuristics (n < 3,000)")
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
				n_jobs=available_cpu_count()
			)

			init = init_method(X)

			Z = TSNEEmbedding(
				init,
				aff,
				learning_rate=lr,
				n_jobs=available_cpu_count(),
				negative_gradient_method="fft"
			)
			Z.optimize(250, exaggeration=12, momentum=0.5, inplace=True, n_jobs=-1)
			Z.optimize(750, exaggeration=exaggeration, momentum=0.8, inplace=True, n_jobs=-1)
		return Z
