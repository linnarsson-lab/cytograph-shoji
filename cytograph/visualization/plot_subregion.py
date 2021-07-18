import matplotlib.patheffects as PathEffects
import logging

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np

import shoji
from cytograph import Module, requires, div0
import numpy_groupies as npg


class PlotSubregion(Module):
	def __init__(self, filename: str = "subregion.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Subregion", "string", ("cells",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotSubregion: Plotting the heatmap")

		clusters = ws.Clusters[:]
		region_names = [
			"Embryo", "Head", "Face", "Brain", "Forebrain", "Telencephalon", "Cortex", "Cortex entorhinal", "Cortex frontotemporal",
			"Cortex hemisphere A", "Cortex hemisphere B", "Cortex lower", "Cortex occipital", "Cortex upper", "Cortical hem", "Hippocampus", "Striatum",
			"Caudate+Putamen", "Subcortex", "Diencephalon", "Hypothalamus", "Thalamus", "Midbrain", "Midbrain dorsal", "Midbrain ventral",
			"Hindbrain", "Metencephalon", "Pons+Medulla", "Medulla", "Cerebellum", "Pons", "Spinal cord", "Spinal cord cervical", "Spinal cord lumbar", "Spinal cord thoracic"
		]
		regions = self.Subregion[:]

		region_dist = np.array([npg.aggregate(clusters, regions == name, func="sum") for name in region_names])
		region_dist = div0(region_dist, region_dist.max(axis=0))

		hsv = np.ones(region_dist.shape + (3,))
		# Set hues for regions
		for ix in range(35):
			hsv[ix, :, 0] = ix / 35
		hsv[..., 1] = region_dist / region_dist.max()  # saturation
		hsv[..., 2] = 1  # value
		rgb = hsv_to_rgb(hsv)

		logging.info(" PlotSubregion: Plotting the manifold")
		fig = plt.figure(figsize=(30, 30))
		ax = fig.add_axes((0, 13 / 15, 1, 2 / 15))
		ax.imshow(rgb, interpolation="none", aspect="auto")
		ax.set_yticks(ticks=range(len(region_names)))
		ax.set_yticklabels(region_names, fontsize=10)
		fig.add_axes((0, 0, 1, 13 / 15))
		xy = ws.Embedding[:]
		n_cells = xy.shape[0]

		marker_size = 2_000_000 / n_cells
		ordering = np.random.permutation(xy.shape[0])
		c = ws.Subregion[:][ordering]
		hsv = np.ones(c.shape + (3,))
		for ix, region in enumerate(region_names):
			hsv[c == region, 0] = ix / 35
		xy = xy[ordering, :]
		plt.scatter(xy[:, 0], xy[:, 1], c=hsv_to_rgb(hsv), s=marker_size, lw=0)
		plt.axis("off")

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=150 if n_cells > 500_000 else 300, bbox_inches='tight')
		plt.close()
