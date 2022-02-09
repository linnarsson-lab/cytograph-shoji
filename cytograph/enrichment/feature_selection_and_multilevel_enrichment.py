from typing import List, Tuple

import numpy as np
import scipy.cluster.hierarchy as hc
import logging
from cytograph import requires, creates, Algorithm
import cytograph as cg
import shoji


class FeatureSelectionAndMultilevelEnrichment(Algorithm):
	def __init__(self, n_genes_per_group: int = 1, mask: List[str] = None, **kwargs) -> None:
		"""
		Args:
			n_genes			Number of genes to select for each cluster group
			mask			Optional list indicating categories of genes that should not be selected
		"""
		super().__init__(**kwargs)
		self.n_genes_per_group = n_genes_per_group
		self.mask = mask if mask is not None else []

	def enrichment_by_clusters(self, ws: shoji.WorkspaceManager) -> np.ndarray:
		"""
		Compute gene enrichment on already aggregated data, by cluster

		Args:
			ws		The workspace
		
		Returns:
			enrichment		A (n_clusters, n_genes) matrix of gene enrichment scores
		"""
		MeanExpression = self.requires["MeanExpression"]

		n_clusters = ws.clusters.length
		x = ws[MeanExpression][:]
		totals = x.sum(axis=1)
		x_norm = (x.T / totals * np.median(totals)).T
		gene_sums = x_norm.sum(axis=0)
		enrichment = np.zeros_like(x_norm)
		for j in range(n_clusters):
			enrichment[j, :] = (x_norm[j, :] + 0.01) / (((gene_sums - x_norm[j, :]) / (n_clusters - 1)) + 0.01)
		return enrichment

	def enrichment_by_cluster_groups(self, ws: shoji.WorkspaceManager, labels: np.ndarray) -> np.ndarray:
		"""
		Compute gene enrichment for aggregate data, by groups of clusters

		Args:
			ws		The workspace
			labels	Labels (0, 1, ...) indicating the cluster groups

		Returns:
			enrichment		A (n_groups, n_genes) matrix of gene enrichment scores
		"""
		n_clusters = labels.max() + 1
		n_cells = self.NCells[:]
		data = (self.MeanExpression[:].T * n_cells).T
		grouped = np.zeros((n_clusters, data.shape[1]))
		for j in range(n_clusters):
			grouped[j, :] = data[labels == j, :].sum(axis=0) / n_cells[labels == j].sum()

		totals = grouped.sum(axis=1)
		x_norm = (grouped.T / totals * np.median(totals)).T
		gene_sums = x_norm.sum(axis=0)
		enrichment = np.zeros_like(x_norm)
		for j in range(n_clusters):
			enrichment[j, :] = (x_norm[j, :] + 0.01) / (((gene_sums - x_norm[j, :]) / (n_clusters - 1)) + 0.01)
		
		return enrichment

	@requires("Species", "string", ())
	@requires("Linkage", "float32", None)
	@requires("MeanExpression", None, ("clusters", "genes"))
	@requires("NCells", "uint64", ("clusters",))
	@requires("Gene", "string", ("genes",))
	@creates("SelectedFeatures", "bool", ("genes",), indices=True)  # indices=True means that the return value is a vector of indices that should be automatically converted to a bool vector
	@creates("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Select genes at multiple levels in a hierarchy of clusters

		Args:
			ws				The shoji workspace containing aggregate data
			save			If true, save the result as tensor SelectedFeatures (default: false)

		Returns:
			SelectedFeatures	Indices of the selected genes
			Enrichment			Enrichment scores
		
		Remarks:
			The number of genes selected will depend on the number of clusters as follows. The dendrogram is
			cut at n = 2, 4, 8, ... (for n <= n_clusters // 2) clusters and the most enriched n_genes are selected for each cluster
			(without replacement). Finally, the most enriched gene in each cluster is selected (without replacement) and added to the list.

			Note that this algorithm requires aggregate tensors (MeanExpression, NCells)
		"""
		# Create symbolic names for the required tensors, which might be renamed by the user
		logging.info(" FeatureSelectionAndMultilevelEnrichment: Selecting features at 2, 4, 8, ... cluster levels")
		n_clusters = ws.clusters.length

		species = cg.Species(self.Species[:])
		mask_genes = species.mask(ws, self.mask)

		n = 2
		selected: List[int] = []
		enrichments: List[np.ndarray] = []
		# Select from the dendrogram
		while n <= n_clusters // 2:
			labels = hc.cut_tree(self.Linkage[:].astype("float64"), n_clusters=n).T[0]
			enr = self.enrichment_by_cluster_groups(ws, labels)
			enrichments.append(enr)
			for j in range(n):
				top = np.argsort(-enr[j, :])
				n_selected = 0
				for t in top:
					if t not in selected and not mask_genes[t]:
						n_selected += 1
						selected.append(t)
						if n_selected == self.n_genes_per_group:
							break
			n *= 2

		logging.info(" FeatureSelectionAndMultilevelEnrichment: Selecting features and calculating enrichment at cluster leaves")
		# Select from the leaves
		enr = self.enrichment_by_clusters(ws)
		enrichments.append(enr)
		for j in range(n_clusters):
			top = np.argsort(-enr[j, :])
			n_selected = 0
			for t in top:
				if t not in selected and not mask_genes[t]:
					n_selected += 1
					selected.append(t)
					if n_selected == self.n_genes_per_group:
						break

		logging.info(f" FeatureSelectionAndMultilevelEnrichment: Selected {len(selected)} features")
		return np.array(selected), np.concatenate(enrichments)
