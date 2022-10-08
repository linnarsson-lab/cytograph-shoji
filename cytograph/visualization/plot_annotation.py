import matplotlib.patheffects as PathEffects
from matplotlib.collections import LineCollection
import logging

import matplotlib.pyplot as plt
import numpy as np

import shoji
from cytograph import Algorithm, requires
from .scatter import scatterc

def indices_to_order_a_like_b(a, b):
	return a.argsort()[b.argsort().argsort()]

class PlotAnnotation(Algorithm):
	def __init__(self, filename: str = "annotation.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	@requires("AnnotationDescription", "string", ("annotations",))
	@requires("AnnotationName", "string", ("annotations",))
	@requires("AnnotationPosterior", "float32", ("clusters","annotations"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotManifold: Plotting the embedding")

		labels = []
		n_cells = self.NCells[:]
		clusters = self.Clusters[:]
		n_clusters = clusters.max() + 1
		cluster_ids = self.ClusterID[:]
		genes = self.Gene[:]

		ordering = indices_to_order_a_like_b(self.ClusterID[:], np.arange(n_clusters))
		ann_desc = self.AnnotationDescription[:]
		ann_names = self.AnnotationName[:]
		ann_post = self.AnnotationPosterior[:].T[:, ordering]


		xy = self.Embedding[:]

		for i in range(ws.clusters.length):
			n = n_cells[cluster_ids == i][0]
			order = ann_post[:,i].argsort()[::-1]
			label = ann_desc[order[:3]]
			label = f"{i:>3} ({n:,} cells) - " + ' | '.join(label)
			labels.append(label)

		plt.figure(figsize=(20, 20))
		ax = plt.subplot(111)
		if "ManifoldIndices" in ws:
			edges = ws.ManifoldIndices[:]
			lc = LineCollection(zip(xy[edges[:, 0]], xy[edges[:, 1]]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
			ax.add_collection(lc)

		MAX_CLUSTERS = 100
		top_clusters = np.argsort(np.bincount(clusters))[-MAX_CLUSTERS:]
		for i in top_clusters:
			pos = np.median(xy[clusters == i], axis=0)
			txt = plt.text(pos[0], pos[1], str(i), size=18, color="black")
			txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

		if n_clusters > MAX_CLUSTERS:
			mask = np.isin(clusters, top_clusters)
			clusters[~mask] = n_clusters
			labels.append(f"{n_clusters} ({n_clusters - MAX_CLUSTERS} clusters not shown)")
		scatterc(xy, c=np.array(labels)[clusters], legend="outside")
		plt.axis("off")

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
		plt.close()