import logging

import matplotlib.pyplot as plt
import numpy as np

import shoji
from cytograph import Algorithm, requires


class PlotDropletClasses(Algorithm):
	def __init__(self, filename: str = "droplets.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Embedding", "float32", ("cells", 2))
	@requires("UnsplicedFraction", "float32", ("cells",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("MitoFraction", "float32", ("cells",))
	@requires("DropletClass", "uint8", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		colors = ["lightgrey", "blue", "orange", "brown", "violet", "red", "lightgreen"]
		class_names = ["Cells", "Large cells", "Doublets", "Cytoplasmic debris", "Cellular debris", "Nuclear debris", "Mitochondrial debris"]

		logging.info(" PlotDropletClasses: Loading droplet classes")
		classes = self.DropletClass[:]
		mt_frac = self.MitoFraction[:]
		u, t = ws.UnsplicedFraction[:], ws.TotalUMIs[:]

		logging.info(" PlotDropletClasses: Plotting")
		plt.figure(figsize=(10, 10))

		plt.subplot(221)
		plt.scatter(u, t, c=mt_frac, cmap="Spectral_r", s=10, lw=0)
		plt.xlabel("Unspliced fraction")
		plt.ylabel("Total UMIs")
		plt.yscale("log")
		plt.ylim(100, 100_000)
		plt.xlim(0, 1)
		plt.colorbar(label="Mitochondrial fraction")

		plt.subplot(222)
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

		plt.subplot(223)
		xy = ws.Embedding[:] if "Embedding" in ws else (ws.Tsne[:] if "Tsne" in ws else np.vstack([ws.X__x[:], ws.X__y[:]]).T)
		plt.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=10, lw=0)
		for i, (color, label) in enumerate(zip(colors, class_names)):
			if 1 == 0:
				continue
			plt.scatter(xy[classes == i, 0], xy[classes == i, 1], s=10, lw=0, label=label, c=color)
		plt.axis("off")

		plt.subplot(224)
		counts = [(classes == i).sum() for i in range(7)]
		plt.pie(counts, colors=colors)
		plt.axis("off")

		passed = classes == 0
		plt.suptitle(f"{ws._name} ({ws.Chemistry[:] if 'Chemistry' in ws else '?'}) | {passed.sum()} of {passed.shape[0]} cells passed ({int(passed.sum() / passed.shape[0] * 100)}%)")
		if save:
			plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
			plt.close()
