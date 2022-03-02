from typing import List, Tuple

import numpy as np
import scipy.cluster.hierarchy as hc
import logging
from cytograph import requires, creates, Algorithm
import cytograph as cg
import shoji
import numpy_groupies as npg


class FeatureSelectionByMultilevelEnrichment(Algorithm):
	def __init__(self, levels_and_genes: Tuple[Tuple[int, int], ...], mask: List[str] = None, **kwargs) -> None:
		"""
		Args:
			levels_and_genes	Tuple like ((2, 10), (5, 5), (100, 1), (None, 1))
			mask				Optional list indicating categories of genes that should not be selected
		
		Remarks:
			The levels_and_genes tuple specifies the levels at which to cut the dendrogram, and the
			number of genes to select per leaf at each level. Each pair (n, i) in the tuple gives
			the number of clusters n to select (or None to select all clusters) and the number
			of genes i per cluster to select.

			After calling fit(), the enrichments for each level are available as self.enrichments
		"""
		super().__init__(**kwargs)
		self.levels_and_genes = levels_and_genes
		self.mask = mask if mask is not None else []

	def enrichment_by_cluster_groups(self, labels: np.ndarray) -> np.ndarray:
		"""
		Compute gene enrichment for aggregate data, by groups of clusters

		Args:
			labels	Labels (0, 1, 0, 2, 3, ...) indicating the cluster groups

		Returns:
			enrichment		A (n_genes, n_groups) matrix of gene enrichment scores
		"""

		n_metaclusters = labels.max() + 1
		n_cells_per_cluster = self.NCells[:].astype("float32")
		metacluster_size = npg.aggregate(labels, n_cells_per_cluster)
		sum_expression = self.MeanExpression[:].T * n_cells_per_cluster
		n_genes = sum_expression.shape[0]

		means = npg.aggregate(labels, sum_expression, axis=1) / metacluster_size

		nnz = npg.aggregate(labels, self.Nonzeros[:].astype("float32"), axis=0).T
		f_nnz = nnz / metacluster_size
		enrichment = np.zeros((n_genes, n_metaclusters))
		for j in range(n_metaclusters):
			ix = np.arange(n_metaclusters) != j
			weights = metacluster_size[ix] / metacluster_size[ix].sum()
			means_other = np.average(means[:, ix], weights=weights, axis=1)
			f_nnz_other = np.average(f_nnz[:, ix], weights=weights, axis=1)
			enrichment[:, j] = (f_nnz[:, j] + 0.1) / (f_nnz_other + 0.1) * (means[:, j] + 0.01) / (means_other + 0.01)

		return enrichment

	@requires("Species", "string", ())
	@requires("Linkage", "float32", None)
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@requires("Nonzeros", "uint64", ("clusters", "genes"))
	@requires("NCells", "uint64", ("clusters",))
	@requires("Gene", "string", ("genes",))
	@creates("SelectedFeatures", "bool", ("genes",), indices=True)  # indices=True means that the return value is a vector of indices that should be automatically converted to a bool vector
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		species = cg.Species(self.Species[:])
		mask_genes = species.mask(ws, self.mask)
		n_clusters = ws.clusters.length
		linkage = self.Linkage[:].astype("float64")

		selected: List[int] = []
		self.enrichments: List[np.ndarray] = []
		# Select from the dendrogram
		for (n, i) in self.levels_and_genes:
			if n is None:
				n = n_clusters
			if n > n_clusters:
				logging.error(f" FeatureSelectionByMultilevelEnrichment: Cannot split {n_clusters} clusters into {n} metaclusters; skipping")
				continue
			logging.info(f" FeatureSelectionByMultilevelEnrichment: Selecting {i * n} features ({i} from each of {n} subtrees)")
			labels = hc.cut_tree(linkage, n_clusters=n).T[0]
			enr = self.enrichment_by_cluster_groups(labels)
			self.enrichments.append(enr)
			for j in range(n):
				top = np.argsort(-enr[:, j])
				n_selected = 0
				for t in top:
					if t not in selected and not mask_genes[t]:
						n_selected += 1
						selected.append(t)
						if n_selected >= i:
							break

		logging.info(f" FeatureSelectionByMultilevelEnrichment: Selected {len(selected)} features")
		return np.sort(np.array(selected))
