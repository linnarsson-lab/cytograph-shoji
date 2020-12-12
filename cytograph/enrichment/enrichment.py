from typing import List, Tuple

import numpy as np
import logging
from cytograph import requires, creates, Module
import cytograph as cg
import shoji


class Enrichment(Module):
	def __init__(self, **kwargs) -> None:
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
		x = self.MeanExpression[...]
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
		"""
		Select genes at multiple levels in a hierarchy of clusters

		Args:
			ws				The shoji workspace containing aggregate data
			save			If true, save the result as tensor SelectedFeatures (default: false)

		Returns:
			Enrichment			Enrichment scores
		"""
		logging.info(" Enrichment: Computing enrichment at cluster leaves")
		enr = self.enrichment_by_clusters(ws)
		return enr
