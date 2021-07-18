import matplotlib.patheffects as PathEffects
import logging

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

import shoji
from cytograph import Module, requires, div0
import numpy_groupies as npg


class PlotRegion(Module):
	def __init__(self, filename: str = "region.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Region", "string", ("cells",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotRegion: Plotting the heatmap")

		clusters = self.Clusters[:]
		region_names = ["Embryo", "Head", "Face", "Brain", "Prosencephalon", "Telencephalon", "Diencephalon", "Mesencephalon", "Rhombencephalon", "Metencephalon", "Myelencephalon", "Spinal cord"]
		regions = ws.Region[:]

		region_dist = np.array([npg.aggregate(clusters, regions == name, func="sum") for name in region_names])
		region_dist = div0(region_dist, region_dist.max(axis=0))

		hsv = np.ones(region_dist.shape + (3,))
		# Set hues for regions
		for ix in range(12):
			hsv[ix, :, 0] = ix / 12
		hsv[..., 1] = region_dist / region_dist.max()  # saturation
		hsv[..., 2] = 1  # value
		rgb = hsv_to_rgb(hsv)

		logging.info(" PlotRegion: Plotting the manifold")
		fig = plt.figure(figsize=(30, 30))
		ax = fig.add_axes((0, 14 / 15, 1, 1 / 15))
		ax.imshow(rgb, interpolation=None, aspect="auto")
		ax.set_yticks(ticks=range(rgb.shape[0]))
		ax.set_yticklabels(region_names, fontsize=12)
		fig.add_axes((0, 0, 1, 14 / 15))
		xy = ws.Embedding[:]
		n_cells = xy.shape[0]

		marker_size = 2_000_000 / n_cells
		ordering = np.random.permutation(xy.shape[0])
		c = ws.Region[:][ordering]
		hsv = np.ones(c.shape + (3,))
		for ix, region in enumerate(region_names):
			hsv[c == region, 0] = ix / 12
		xy = xy[ordering, :]
		plt.scatter(xy[:, 0], xy[:, 1], c=hsv_to_rgb(hsv), s=marker_size, lw=0)
		plt.axis("off")

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=150 if n_cells > 500_000 else 300, bbox_inches='tight')
		plt.close()
