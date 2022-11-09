import logging

import matplotlib.pyplot as plt
import numpy_groupies as npg
import numpy as np

import shoji
from cytograph import Algorithm, requires
from cytograph.visualization.colors import colorize


class PlotDropletClasses(Algorithm):
	def __init__(self, **kwargs) -> None:
		super().__init__()
		if "filename" in kwargs:
			self.filename = kwargs["filename"]
		else:
			self.filename = "droplets.png"
		self.kwargs = kwargs

	@requires("Embedding", "float32", ("cells", 2))
	@requires("UnsplicedFraction", "float32", ("cells",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("MitoFraction", "float32", ("cells",))
	@requires("DropletClass", "uint8", ("cells",))
	@requires("Clusters", "uint32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		colors = ["lightgrey", "blue", "orange", "brown", "violet", "red", "lightgreen"]
		class_names = ["Cells", "Large cells", "Doublets", "Cytoplasmic debris", "Cellular debris", "Nuclear debris", "Mitochondrial debris"]

		logging.info(" PlotDropletClasses: Loading droplet classes")
		classes = self.DropletClass[:]
		mt_frac = self.MitoFraction[:]
		u, t = ws.UnsplicedFraction[:], ws.TotalUMIs[:]

		logging.info(" PlotDropletClasses: Plotting")
		plt.figure(figsize=(10, 10))

		plt.subplot(3, 2, 1)
		plt.scatter(u, t, c=mt_frac, cmap="Spectral_r", s=10, lw=0)
		if "min_unspliced_fraction" in self.kwargs:
			plt.vlines(self.kwargs["min_unspliced_fraction"], 100, 100_000, linestyles='dashed', lw=1, color="green")
		if "min_umis" in self.kwargs:
			plt.hlines(self.kwargs["min_umis"], 0, 1, linestyles='dashed', lw=1, color="green")
		if "m" in self.kwargs and "k" in self.kwargs:
			m = self.kwargs["m"]
			k = self.kwargs["k"]
			plt.plot([0, 1], [m, m + k], color="green", ls="dashed", lw=1)
		plt.xlabel("Unspliced fraction")
		plt.ylabel("Total UMIs")
		plt.yscale("log")
		plt.ylim(100, 100_000)
		plt.xlim(0, 1)

		plt.colorbar(label="Mitochondrial fraction")

		plt.subplot(3,2,2)
		plt.scatter(u, t, c="lightgrey", cmap=plt.cm.tab10, s=10, lw=0)
		for i, (color, label) in enumerate(zip(colors, class_names)):
			if 1 == 0:
				continue
			plt.scatter(u[classes == i], t[classes == i], s=10, lw=0, c=color, label=label)
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=3)
		plt.xlabel("Unspliced UMI fraction")
		plt.ylabel("Total UMIs")
		plt.yscale("log")
		plt.ylim(100, 100_000)
		plt.xlim(0, 1)

		plt.subplot(3,2,3)
		xy = ws.Embedding[:] if "Embedding" in ws else (ws.Tsne[:] if "Tsne" in ws else np.vstack([ws.X__x[:], ws.X__y[:]]).T)
		plt.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=10, lw=0)
		for i, (color, label) in enumerate(zip(colors, class_names)):
			if 1 == 0:
				continue
			plt.scatter(xy[classes == i, 0], xy[classes == i, 1], s=10, lw=0, label=label, c=color)
		plt.axis("off")

		plt.subplot(3,2,4)
		counts = [(classes == i).sum() for i in range(7)]
		plt.pie(counts, colors=colors)
		plt.axis("off")


		plt.subplot(3,2,5)
		clusters = ws.Clusters[:]
		color_clust = colorize(clusters)
		MAX_CLUSTERS = 100
		top_clusters = np.argsort(np.bincount(clusters))[-MAX_CLUSTERS:]

		for i in top_clusters:
			pos = np.median(xy[clusters == i], axis=0)
			txt = plt.text(pos[0], pos[1], str(i), size=10, color="black")
		plt.scatter(xy[:, 0], xy[:, 1], c=color_clust, s=10, lw=0, alpha=0.1)
		plt.axis('off')


		dropind = np.unique(classes)
		drop_attr = classes
		cluster_id, total = np.unique(self.Clusters[:], return_counts=True)

		plt.subplot(3,2,6)
		bottom_counter = np.zeros(len(cluster_id))
		for i in dropind[::-1]:
			counts = npg.aggregate(self.Clusters[:], drop_attr == i)
			counts = (counts / total) * 100
			plt.bar(range(len(counts)), counts, bottom=bottom_counter, color=[colors[i]]*len(counts))
			bottom_counter += counts

		plt.xticks(np.arange(0, len(counts), 1), rotation=90)
		plt.xlabel('Cluster ID')
		plt.ylabel('Fraction of droplet class (%)')

		passed = classes == 0
		plt.suptitle(f"{ws._name} ({ws.Chemistry[:] if 'Chemistry' in ws else '?'}) | {passed.sum()} of {passed.shape[0]} cells passed ({int(passed.sum() / passed.shape[0] * 100)}%)")
		plt.tight_layout(pad=1.2, h_pad=2.3, w_pad=2.5)



		passed = classes == 0
		plt.suptitle(f"{ws._name} ({ws.Chemistry[:] if 'Chemistry' in ws else '?'}) | {passed.sum()} of {passed.shape[0]} cells passed ({int(passed.sum() / passed.shape[0] * 100)}%)")
		if save:
			plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
			plt.close()

