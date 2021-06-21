import logging
import shoji
import numpy as np
from cytograph import creates, requires, Module
from cytograph.pipeline import Config
from scipy.cluster.hierarchy import cut_tree


class CutDendrogram(Module):
	def __init__(self, n_trees: int = 2, split_when_over: int = 50, **kwargs) -> None:
		"""
		Cut the dendrogram of clusters and return labels for subtrees. 

		Remarks:
			For any subtree with more than split_when_over clusters, a new punchcard is created
		"""
		super().__init__(**kwargs)
		self.n_trees = n_trees
		self.split_when_over = split_when_over

	@requires("Linkage", "float32", (None, 4))
	@requires("Clusters", "uint32", ("cells",))
	@creates("Subtree", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool) -> np.ndarray:
		config = Config.load()
		punchards_path = config["paths"]["build"] / "punchcards"
		punchcard = config["punchcard"]

		logging.info(f" CutDendrogram: Cutting to create {self.n_trees} subtrees")
		z = self.Linkage[:].astype("float64")
		cuts = cut_tree(z, n_clusters=self.n_trees).T[0]
		clusters = self.Clusters[:]
		subtrees = np.zeros_like(clusters)
		for ix, tree in enumerate(cuts):
			subtrees[clusters == ix] = tree
		for ix in range(self.n_trees):
			n_clusters = len(np.unique(clusters[subtrees == ix]))
			if n_clusters > self.split_when_over:
				new_name = punchcard.name + "ABCDEFGHIJKLMNOPQRSTUVXYZ"[ix]
				logging.info(f" CutDendrogram: Creating punchcard {new_name} from {n_clusters} clusters of branch {ix}")
				with open(punchards_path / (new_name + ".yaml"), "w") as f:
					f.write(f'''
onlyif: "ws.Subtree == {ix}"

recipe: {punchcard.recipe}

resources:
  n_cpus: {punchcard.resources["n_cpus"]}
  n_gpus: {punchcard.resources["n_gpus"]}
  memory: {punchcard.resources["memory"]}

sources: [{punchcard.name}]
''')
		return subtrees
