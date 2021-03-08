import logging
import shoji
import numpy as np
from cytograph import creates, requires, Module
from scipy.cluster.hierarchy import cut_tree


class CutDendrogram(Module):
	def __init__(self, n_trees: int = 2, **kwargs) -> None:
		"""
		Cut the dendrogram of clusters and return labels for subtrees
		"""
		super().__init__(**kwargs)
		self.n_trees = n_trees

	@requires("Linkage", "float32", (None, 4))
	@requires("Clusters", "uint32", ("cells",))
	@creates("Subtree", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool) -> np.ndarray:
		logging.info(f" CutDendrogram: Cutting to create {self.n_trees} subtrees")
		z = self.Linkage[:].astype("float64")
		cuts = cut_tree(z, n_clusters=self.n_trees).T[0]
		clusters = self.Clusters[:]
		subtrees = np.zeros_like(clusters)
		for ix, tree in enumerate(cuts):
			subtrees[clusters == ix] = tree
		return subtrees
