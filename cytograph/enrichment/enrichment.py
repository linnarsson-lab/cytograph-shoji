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

	def enrichment_by_clusters(self, ws: shoji.WorkspaceManager) -> np.ndarray:
		"""
		Compute gene enrichment on already aggregated data, by cluster

		Args:
			ws		The workspace
		
		Returns:
			enrichment		A (n_clusters, n_genes) matrix of gene enrichment scores
		"""
		n_clusters = ws.clusters.length
		x = self.MeanExpression[:]
		totals = x.sum(axis=1)
		x_norm = (x.T / totals * np.median(totals)).T
		gene_sums = x_norm.sum(axis=0)
		enrichment = np.zeros_like(x_norm)
		for j in range(n_clusters):
			enrichment[j, :] = (x_norm[j, :] + 0.01) / (((gene_sums - x_norm[j, :]) / (n_clusters - 1)) + 0.01)
		return enrichment

	@requires("MeanExpression", None, ("clusters", "genes"))
	@creates("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:

		logging.info(" Enrichment: Computing enrichment at cluster leaves")
		enr = self.enrichment_by_clusters(ws)
		return enr
