import logging

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

import shoji
from cytograph import Algorithm, requires, div0
import numpy_groupies as npg


class PlotAge(Algorithm):
	def __init__(self, filename: str = "age.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Age", "float32", ("cells",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotAge: Plotting the heatmap")

		ages_raw = self.Age[:]
		ages = ages_raw.astype("int")
		clusters = ws.Clusters[:]
		group_idx = np.vstack([ages, clusters])

		age_dist = npg.aggregate(group_idx, ages, func="count", size=(15, clusters.max()+1))
		age_dist = div0(age_dist.T, age_dist.sum(axis=1)).T
		age_dist = div0(age_dist, age_dist.max(axis=0))

		fig = plt.figure(figsize=(30, 30))
		ax = fig.add_axes((0, 14 / 15, 1, 1 / 15))

		hsv = np.ones(age_dist.shape + (3,))
		# Set hues for ages
		for ix in range(age_dist.shape[0]):
			hsv[ix, :, 0] = 0.75 - ix / 20
		hsv[..., 1] = age_dist / age_dist.max()  # saturation
		hsv[..., 2] = 1  # value
		rgb = hsv_to_rgb(hsv)

		ax.imshow(rgb, cmap="Greys", interpolation="none", aspect="auto", origin="upper")
		ax.set_yticks(ticks=range(15))
		ax.set_yticklabels(labels=range(15), fontsize=10)
		ax.set_ylabel("Age (p.c.w.)", fontsize=14)

		logging.info(" PlotAge: Plotting the manifold")
		fig.add_axes((0, 0, 1, 14 / 15))
		xy = self.Embedding[:]
		n_cells = xy.shape[0]
		
		marker_size = 2_000_000 / n_cells
		ordering = np.random.permutation(xy.shape[0])
		c = ages_raw[ordering]
		hsv = np.ones(c.shape + (3,))
		hsv[:, 0] = 0.75 - c / 20
		xy = xy[ordering, :]
		plt.scatter(xy[:, 0], xy[:, 1], c=hsv_to_rgb(hsv), s=marker_size, lw=0)
		plt.axis("off")

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=150 if n_cells > 500_000 else 300, bbox_inches='tight')
		plt.close()
