from typing import List, Tuple

import numpy as np
import logging
from cytograph import requires, creates, Algorithm
import cytograph as cg
import shoji


class Enrichment(Algorithm):
	def __init__(self, **kwargs) -> None:
		"""
		Compute gene enrichment in clusters

		Remarks:
			Gene enrichment is computed as the regularized fold-change between
			each cluster and all other clusters. This is different from the
			previous cytograph version, which also considered the fraction non-zeros.
		"""
		super().__init__(**kwargs)

	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@requires("NCells", "uint64", ("clusters",))
	@requires("Nonzeros", "uint64", ("clusters", "genes"))
	@creates("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		logging.info(" Enrichment: Computing enrichment by cluster")

		n_clusters = ws.clusters.length
		cluster_size = self.NCells[:]
		means = self.MeanExpression[:]
		# totals = x.sum(axis=1)
		# means = (x.T / totals * np.median(totals)).T
		nnz = self.Nonzeros[:]
		f_nnz = nnz / cluster_size
		enrichment = np.zeros_like(means)
		for j in range(n_clusters):
			ix = np.arange(n_clusters) != j
			weights = cluster_size[ix] / cluster_size[ix].sum()
			means_other = np.average(means[:, ix], weights=weights, axis=1)
			f_nnz_other = np.average(f_nnz[:, ix], weights=weights, axis=1)
			enrichment[:, j] = (f_nnz[:, j] + 0.1) / (f_nnz_other + 0.1) * (means[:, j] + 0.01) / (means_other + 0.01)
		return enrichment
