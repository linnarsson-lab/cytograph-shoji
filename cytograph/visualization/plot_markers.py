import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import shoji
from cytograph import Module, requires
from cytograph.pipeline import Config
import yaml


class PlotMarkers(Module):
	def __init__(self, markers: str = "immune", filename: str = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename if filename is not None else markers + ".png"
		self.markers = markers

	@requires("Gene", "string", ("genes",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotSubregion: Plotting the heatmap")
		
		config = Config.load()
		with open(config.path / "markers.yaml") as f:
			markers: List[Tuple[str, str]] = yaml.load(f, Loader=yaml.SafeLoader)[self.markers]

		# Load data
		genes = self.Gene[:]
		ordering = np.argsort(self.ClusterID[:])
		mean_x = self.MeanExpression[:][ordering]

		m = []
		m_names = []
		for gene, desc in markers:
			gene_ix = np.where(genes == gene)[0][0]
			m.append(mean_x[:, gene_ix])
			m_names.append(gene + " (" + desc + ")")
		x = np.array(m)

		# Normalize
		totals = mean_x.sum(axis=1)
		x_norm = (x / totals * np.median(totals)).T

		fig = plt.figure(figsize=(30, 30))
		strip_height = len(markers) / 10
		ax = fig.add_axes((0, (15 - strip_height) / 15, 1, strip_height / 15))
		ax.imshow(np.log10(x_norm.T + 0.001), vmin=-1, vmax=2, cmap="RdGy_r", interpolation="none", aspect="auto")
		ax.set_yticks(ticks=range(len(m_names)))
		ax.set_yticklabels(m_names, fontsize=10)

		xy = self.Embedding[:]
		n_cells = xy.shape[0]
		for ix in np.arange(len(markers))[::-1]:
			(gene, desc) = markers[ix]
			gene_ix = np.where(genes == gene)[0][0]
			ax = fig.add_axes(((ix % 5) * 0.2, (ix // 5) * 0.2 * (15 - strip_height) / 15, 0.2, (15 - strip_height) / 15 / 5))
			x = ws.Expression[:, gene_ix].T[0]
			ax.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=100_000 / n_cells)
			ax.scatter(xy[x > 0, 0], xy[x > 0, 1], c=x[x > 0], s=100_000 / n_cells)
			ax.set_title(gene + "\n" + desc, fontsize=14)
			plt.axis("off")

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=150 if n_cells > 500_000 else 300, bbox_inches='tight')
		plt.close()
