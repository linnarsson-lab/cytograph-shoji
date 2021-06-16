import matplotlib.patheffects as PathEffects
import logging

import matplotlib.pyplot as plt
import numpy as np

import shoji
from cytograph import Module, requires
from .scatter import scatterc


class PlotEmbedding(Module):
	def __init__(self, filename: str = "embedding.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotManifold: Plotting the embedding")

		labels = []
		n_cells = self.NCells[:]
		clusters = self.Clusters[:]
		cluster_ids = self.ClusterID[:]
		genes = self.Gene[:]
		enrichment = self.Enrichment[:]
		for i in range(ws.clusters.length):
			n = n_cells[cluster_ids == i][0]
			label = f"{i:>3} ({n:,} cells) "
			label += " ".join(genes[np.argsort(-enrichment[cluster_ids == i, :][0])[:10]])
			labels.append(label)
		plt.figure(figsize=(20, 20))
		xy = self.Embedding[:]
		scatterc(xy, c=np.array(labels)[clusters], legend="outside")
		for i in range(ws.clusters.length):
			pos = np.median(xy[clusters == i], axis=0)
			txt = plt.text(pos[0], pos[1], str(i), size=14)
			txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
		plt.axis("off")

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300)
		plt.close()
