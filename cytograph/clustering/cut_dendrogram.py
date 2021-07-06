import logging
import shoji
import numpy as np
from cytograph import creates, requires, Module
from cytograph.pipeline import Config
from scipy.cluster.hierarchy import cut_tree
from typing import Dict


class CutDendrogram(Module):
	def __init__(self, n_trees: int = 2, split_when_over: int = 50, split_with_settings: Dict = None, **kwargs) -> None:
		"""
		Cut the dendrogram of clusters and return labels for subtrees. 

		Remarks:
			For any subtree with more than split_when_over clusters, a new punchcard is created
		"""
		super().__init__(**kwargs)
		self.n_trees = n_trees
		self.split_when_over = split_when_over
		self.split_with_settings = split_with_settings if split_with_settings is not None else {}

	@requires("Linkage", "float32", (None, 4))
	@requires("Clusters", "uint32", ("cells",))
	@creates("Subtree", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool) -> np.ndarray:
		config = Config.load()
		punchards_path = config.path / "punchcards"
		punchcard = config.punchcard
		assert punchcard is not None

		logging.info(f" CutDendrogram: Cutting to create {self.n_trees} subtrees")
		z = self.Linkage[:].astype("float64")
		cuts = cut_tree(z, n_clusters=self.n_trees).T[0]
		clusters = self.Clusters[:]
		n_clusters = len(np.unique(clusters))
		subtrees = np.zeros_like(clusters)
		for ix, tree in enumerate(cuts):
			subtrees[clusters == ix] = tree
		if n_clusters > self.split_when_over:
			for ix in range(self.n_trees):
				new_name = punchcard.name + "ABCDEFGHIJKLMNOPQRSTUVXYZ"[ix]
				logging.info(f" CutDendrogram: Creating punchcard '{new_name}'")
				recipe = self.split_with_settings.get("recipe", punchcard.recipe)
				n_cpus = self.split_with_settings.get("n_cpus", punchcard.resources.n_cpus)
				n_gpus = self.split_with_settings.get("n_gpus", punchcard.resources.n_gpus)
				memory = self.split_with_settings.get("memory", punchcard.resources.memory)

				with open(punchards_path / (new_name + ".yaml"), "w") as f:
					f.write(f'''
onlyif: "ws.Subtree == {ix}"

recipe: {recipe}

resources:
  n_cpus: {n_cpus}
  n_gpus: {n_gpus}
  memory: {memory}

sources: [{punchcard.name}]
''')
		return subtrees
