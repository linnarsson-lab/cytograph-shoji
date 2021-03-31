from typing import Dict, List
import logging

import matplotlib.pyplot as plt
import numpy as np

import shoji
from cytograph import Module, requires
from .scatter import scatterc, scattern


class PlotBatches(Module):
	def __init__(self, filename: str = "batches.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename
	
	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotBatches: Loading the embedding")
		xy = self.Embedding[:]

		logging.info(" PlotBatches: Plotting")
		n_cells = ws.cells.length
		marker_size = 1000 / np.sqrt(n_cells)
		plt.figure(figsize=(16, 14))

		def expand_scalars(x):
			if x.ndim == 0:
				return np.full(n_cells, fill_value=x)
			else:
				return x

		# Age
		age = expand_scalars(ws.Age[:])
		plt.subplot(2, 2, 1)
		plt.scatter(xy[:, 0], xy[:, 1], c=age, s=marker_size, lw=0, cmap="rainbow")
		plt.colorbar(fraction=0.02, pad=0.04)
		plt.axis("off")
		plt.title(f"Age")

		# Tissue
		tissue = expand_scalars(ws.Tissue[:])
		plt.subplot(2, 2, 2)
		scatterc(xy, c=tissue, s=marker_size, lw=0)
		plt.axis("off")
		plt.title(f"Tissue")

		# Chemistry
		chemistry = expand_scalars(ws.Chemistry[:])
		plt.subplot(2, 2, 3)
		scatterc(xy, c=chemistry, s=marker_size, lw=0)
		plt.axis("off")
		plt.title(f"Chemistry")

		# Chemistry
		sid = expand_scalars(ws.SampleID[:])
		plt.subplot(2, 2, 4)
		scatterc(xy, c=sid, s=marker_size, lw=0)
		plt.axis("off")
		plt.title(f"SampleID")

		plt.savefig(self.export_dir / self.filename, dpi=300)
		plt.close()
