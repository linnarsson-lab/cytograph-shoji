from typing import Tuple, List

import numpy as np

import loompy


class AutoAutoAnnotator:
	"""
	Automatically discover suitable auto-annotation marker combinations
	"""
	def __init__(self, pep: float = 0.05, n_genes: int = 6, genes_allowed: np.ndarray = None) -> None:
		self.pep = pep
		self.n_genes = max(2, n_genes)
		self.genes_allowed = genes_allowed
	
	def fit(self, dsagg: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Find highly specific and robust auto-annotation gene sets for all clusters in the file

		Returns:
			selected		The selected genes (indexes into the rows), shape (n_genes, n_clusters)
			selectivity		The cumulative selectivity (number of clusters identified), shape (n_genes, n_clusters)
			specificity		The cumulative specificity (difference in probabililty of identifying the cluster,
							relative to the second likeliest cluster), shape (n_genes, n_clusters)
			robustness		The cumulative robustness (probability of identifying the cluster), shape (n_genes, n_clusters)
		"""
		blocked = np.in1d(dsagg.ra.Gene, ['Xist', 'Tsix', 'Junb', 'Fos', 'Egr1', 'Jun']).nonzero()[0]

		trinaries = dsagg.layer["trinaries"][:, :]
		enrichment = dsagg.layer["enrichment"][:, :]
		n_clusters = dsagg.shape[1]
		positives = (trinaries > (1 - self.pep)).astype('int')
		genes = np.where(np.logical_and(positives.sum(axis=1) < n_clusters * 0.5, positives.sum(axis=1) > 0))[0]
		if self.genes_allowed is not None:
			genes = np.intersect1d(genes, self.genes_allowed)
	
		# Select the most enriched gene in each cluster
		gene1 = []  # type: List[int]
		for ix in range(dsagg.shape[1]):
			candidates = np.where(positives[:, ix] == 1)[0]
			candidates = np.setdiff1d(candidates, blocked)
			ordering = np.argsort(-enrichment[candidates, ix])
			try:
				gene1.append(candidates[ordering][0])
			except IndexError:
				gene1.append(0)  # NOTE NOTE NOTE very bad patch but I want to make it run to the end
		selected = np.array(gene1)[None, :]

		# Select the most enriched most specific gene for each cluster, given genes previously selected
		for _ in range(self.n_genes - 1):
			gene2 = []
			for ix in range(dsagg.shape[1]):
				# For each gene, the number of clusters where it's positive, shape (n_genes)
				breadth = (positives * np.prod(positives[selected[:, ix]], axis=0)).sum(axis=1)
				# The genes that are expressed in cluster ix, excluding previously selected and blocked genes
				candidates = np.where(positives[:, ix] == 1)[0]
				candidates = np.setdiff1d(candidates, selected)
				candidates = np.setdiff1d(candidates, blocked)
				try:
					# Now select the most specific gene, ranked by enrichment
					narrowest = breadth[candidates][breadth[candidates].nonzero()].min()
					candidates = np.intersect1d(candidates, np.where(breadth == narrowest)[0])
					ordering = np.argsort(-enrichment[candidates, ix])
					gene2.append(candidates[ordering][0])
				except (IndexError, ValueError):
					gene2.append(0)  # NOTE NOTE NOTE very bad patch but I want to make it run to the end
			gene2 = np.array(gene2)
			selected = np.vstack([selected, gene2])

		selectivity = np.cumprod(positives[selected], axis=0).sum(axis=1)
		robustness = np.array([np.cumprod(trinaries[selected[:, ix], ix]) for ix in np.arange(n_clusters)]).T

		specificity = []
		for c in np.arange(n_clusters):
			a = np.cumprod(trinaries[selected[:, :], c], axis=0)
			vals = []
			for ix in np.arange(5):
				temp = np.sort(a[ix, :])[-2:]
				vals.append(temp[-1] - temp[-2])
			specificity.append(vals)
		specificity = np.array(specificity).T

		return (selected, selectivity, specificity, robustness)

	def annotate(self, ds: loompy.LoomConnection) -> None:
		"""
		Annotate the loom file with marker gene sets based on auto-auto-annotation

		Remarks:
			Creates the following column attributes:
				MarkerGenes			Space-separated list of six marker genes
				MarkerSelectivity	Space-separated list of cumulative selectivity
				MarkerSpecificity	Space-separated list of cumulative specificity
				MarkerRobustness	Space-separated list of cumulative robustness
			
			See the fit() method for definitions of the metrics
		"""
		(selected, selectivity, specificity, robustness) = self.fit(ds)
		n_clusters = ds.ca.Clusters.max() + 1
		ds.ca.MarkerGenes = [" ".join(ds.ra.Gene[selected[:, ix]]) for ix in np.arange(n_clusters)]
		ds.ca.MarkerSelectivity = [" ".join([str(x) for x in selectivity[:, ix]]) for ix in np.arange(n_clusters)]
		ds.ca.MarkerSpecificity = [" ".join([f"{x:.2}" for x in specificity[:, ix]]) for ix in np.arange(n_clusters)]
		ds.ca.MarkerRobustness = [" ".join([f"{x:.2}" for x in robustness[:, ix]]) for ix in np.arange(n_clusters)]
