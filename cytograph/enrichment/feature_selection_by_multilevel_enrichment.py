from typing import List, Tuple

import numpy as np
import scipy.cluster.hierarchy as hc
import logging
from cytograph import CytographMethod
import shoji


class FeatureSelectionByMultilevelEnrichment(CytographMethod):
	def __init__(self) -> None:
		self._requires = [
			("Linkage", "float64", None),
			("Expression", None, ("cells", "genes")),
			("NCells", "int64", ("clusters",)),
			("Gene", "string", ("genes",))
		]

	def enrichment_by_clusters(self, ws: shoji.WorkspaceManager) -> np.ndarray:
		"""
		Compute gene enrichment for a workspace of aggregate data, by cluster

		Args:
			ws		The workspace
		
		Returns:
			enrichment		A (n_clusters, n_genes) matrix of gene enrichment scores
		"""
		n_clusters = ws.clusters.length
		view = ws[:]

		totals = view.Expression.sum(axis=1)
		x_norm = (view.Expression.T / totals * np.median(totals)).T
		gene_sums = x_norm.sum(axis=0)
		enrichment = np.zeros_like(x_norm)
		for j in range(n_clusters):
			enrichment[j, :] = (x_norm[j, :] + 0.01) / (((gene_sums - x_norm[j, :]) / (n_clusters - 1)) + 0.01)
		return enrichment

	def enrichment_by_cluster_groups(self, ws: shoji.WorkspaceManager, labels: np.ndarray) -> np.ndarray:
		"""
		Compute gene enrichment for a workspace of aggregate data, by groups of clusters

		Args:
			ws		The workspace
			labels	Labels (0, 1, ...) indicating the cluster groups

		Returns:
			enrichment		A (n_groups, n_genes) matrix of gene enrichment scores
		"""
		n_clusters = labels.max() + 1
		view = ws[:]
		data = (view.Expression.T * view.NCells).T
		grouped = np.zeros((n_clusters, data.shape[1]))
		for j in range(n_clusters):
			grouped[j, :] = data[labels == j, :].sum(axis=0) / view.NCells[labels == j].sum()

		totals = grouped.sum(axis=1)
		x_norm = (grouped.T / totals * np.median(totals)).T
		gene_sums = x_norm.sum(axis=0)
		enrichment = np.zeros_like(x_norm)
		for j in range(n_clusters):
			enrichment[j, :] = (x_norm[j, :] + 0.01) / (((gene_sums - x_norm[j, :]) / (n_clusters - 1)) + 0.01)
		
		return enrichment

	def fit(self, ws: shoji.WorkspaceManager, preselected: List[str] = [], mask: np.ndarray = None):
		"""
		Select genes at multiple levels in a hierarchy of clusters

		Args:
			ws				The shoji workspace containing aggregate data
			preselected		Optional list of preselected genes (names as strings)
			mask			Bool mask indicating genes that should not be selected

		Returns:
			selected		np.ndarray of ints giving indices into the genes dimension for the selected genes
		
		Remarks:
			The number of genes selected will depend on the number of clusters as follows. The dendrogram is
			cut at n = 2, 4, 8, ... (for n <= n_clusters // 2) clusters and the most enriched gene is selected for each cluster (without
			replacement). Finally, the most enriched gene in each cluster is selected (without replacement) and added to the list.
		"""
		logging.info("FeatureSelectionByMultilevelEnrichment: Selecting features at 2, 4, 8, ... cluster levels")
		n_clusters = ws.clusters.length

		n = 2
		selected: List[int] = []
		genes = ws[:].Gene
		for gene in preselected:
			if gene in genes:
				selected.append(np.where(genes == gene)[0][0])

		# Select from the dendrogram
		while n <= n_clusters // 2:
			labels = hc.cut_tree(ws[:].Linkage, n_clusters=n).T[0]
			enr = self.enrichment_by_cluster_groups(ws, labels)
			for j in range(n):
				top = np.argsort(-enr[j, :])
				for t in top:
					if t not in selected and (mask is not None and not mask[t]):
						selected.append(t)
						break
			n *= 2

		logging.info("FeatureSelectionByMultilevelEnrichment: Selecting features at cluster leaves")
		# Select from the leaves
		enr = self.enrichment_by_clusters(ws)
		for j in range(n_clusters):
			top = np.argsort(-enr[j, :])
			for t in top:
				if t not in selected and (mask is not None and not mask[t]):
					selected.append(t)
					break
		logging.info(f"FeatureSelectionByMultilevelEnrichment: Done; {len(selected)} features selected")
		return selected

	def fit_save(self, ws: shoji.WorkspaceManager, preselected: List[str] = [], mask: np.ndarray = None):
		selected_list = self.fit(ws, preselected, mask)
		logging.info("FeatureSelectionByMultilevelEnrichment: Saving selected features as bool tensor 'SelectedFeatures'")
		selected_bool = np.zeros(ws.genes.length, dtype="bool")
		selected_bool[selected_list] = True
		ws.SelectedFeatures = shoji.Tensor("bool", ("genes",), inits=selected_bool)
		return selected_list
