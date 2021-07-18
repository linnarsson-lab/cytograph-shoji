import logging

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np

import shoji
from cytograph import Module, requires, div0
import numpy_groupies as npg


class PlotCellCycle(Module):
	def __init__(self, filename: str = "cellcycle.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("CellCycleFraction", "string", ("cells",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotCellCycle: Plotting the heatmap")

		clusters = self.Clusters[:]
		n_clusters = clusters.max() + 1
		cc_raw = self.CellCycleFraction[:]

		cc = np.clip((cc_raw * 1000).astype("int"), 0, 19)
		group_idx = np.vstack([cc, clusters])

		cc_dist = npg.aggregate(group_idx, cc, func="count", size=(20, n_clusters))
		cc_dist = div0(cc_dist.T, cc_dist.sum(axis=1)).T
		cc_dist = div0(cc_dist, cc_dist.max(axis=0))

		fig = plt.figure(figsize=(30, 30))
		ax = fig.add_axes((0, 14 / 15, 1, 1 / 15))

		ax.imshow(cc_dist, cmap="Greys", interpolation=None, aspect="auto", origin="lower")
		ax.set_yticks(ticks=(0, 3.5, 9.5, 19.5))
		ax.set_yticklabels(labels=["0", "0.4%", "1%", "2%"], fontsize=15)
		ax.set_ylabel("Cell cycle (% UMIs)", fontsize=18)
		ax.get_yticklabels()[1].set_color("red")
		ax.hlines(3.5, -0.5, n_clusters - 0.5, colors="red", lw=1, linestyles="dashed")

		logging.info(" PlotCellCycle: Plotting the manifold")
		ax = fig.add_axes((0, 0, 1, 14 / 15))
		xy = ws.Embedding[:]
		n_cells = xy.shape[0]
		marker_size = 2_000_000 / n_cells
		ordering = np.random.permutation(xy.shape[0])
		xy_ = xy[ordering, :]
		ax.scatter(xy_[:, 0], xy_[:, 1], c="lightgrey", s=marker_size, lw=0)
		cycling = cc_raw[ordering] > 0.004
		ax.scatter(xy_[cycling, 0], xy_[cycling, 1], c=cc_raw[cycling], cmap="viridis", s=marker_size, lw=0)

		top_clusters = np.argsort(np.bincount(clusters))[-100:]
		for i in top_clusters:
			pos = np.median(xy[clusters == i], axis=0)
			txt = ax.text(pos[0], pos[1], str(i), size=32, color="deeppink")
			txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
		plt.axis("off")

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=150 if n_cells.sum() > 500_000 else 300, bbox_inches='tight')
		plt.close()
