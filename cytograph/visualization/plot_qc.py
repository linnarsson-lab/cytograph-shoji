from typing import Dict, List
import logging

import matplotlib.pyplot as plt
import numpy as np

import shoji
from cytograph import Module, requires


class PlotQC(Module):
	def __init__(self, filename: str = "qc.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename
	
	@requires("Embedding", "float32", ("cells", 2))
	@requires("ValidCells", "bool", ("cells",))
	@requires("DoubletFlag", "bool", ("cells",))
	@requires("DoubletScore", "float32", ("cells",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("NGenes", "uint32", ("cells",))
	@requires("UnsplicedFraction", "float32", ("cells",))
	@requires("MitoFraction", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotQC: Loading the embedding")
		xy = self.Embedding[:]

		logging.info(" PlotQC: Plotting")
		n_cells = ws.cells.length
		marker_size = 1000 / np.sqrt(n_cells)
		plt.figure(figsize=(16, 20))

		# Invalid cells
		plt.subplot(3, 2, 1)
		plt.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=marker_size, lw=0)
		selected = self.ValidCells[:]
		plt.scatter(xy[~selected, 0], xy[~selected, 1], c="crimson", s=marker_size, lw=0)
		plt.axis("off")
		plt.title(f"Invalid cells ({(~selected).sum():,} of {n_cells:,} = {int((~selected).sum() / n_cells * 100)}%)")

		# Doublets
		plt.subplot(3, 2, 2)
		plt.scatter(xy[:, 0], xy[:, 1], c=self.DoubletScore[:], s=marker_size, lw=0, cmap="cividis")
		plt.colorbar(fraction=0.02, pad=0.04)
		selected = self.DoubletFlag[:]
		plt.scatter(xy[selected, 0], xy[selected, 1], fc="none", ec="crimson", s=marker_size * 2, lw=1)
		plt.axis("off")
		plt.title("DoubletScore (color), DoubletFlag (red outline)")

		# UMIs
		plt.subplot(3, 2, 3)
		plt.scatter(xy[:, 0], xy[:, 1], c=np.log10(self.TotalUMIs[:]), s=marker_size, lw=0, cmap="rainbow")
		plt.axis("off")
		plt.title("Total UMIs (log10 scale)")
		plt.colorbar(fraction=0.02, pad=0.04)

		# Genes
		plt.subplot(3, 2, 4)
		plt.scatter(xy[:, 0], xy[:, 1], c=np.log10(self.NGenes[:]), s=marker_size, lw=0, cmap="rainbow")
		plt.axis("off")
		plt.title("Total Genes (log10 scale)")
		plt.colorbar(fraction=0.02, pad=0.04)

		# Unspliced
		plt.subplot(3,2,5)
		uf = self.UnsplicedFraction[:]
		plt.scatter(xy[:, 0], xy[:, 1], c=uf * 100, s=marker_size, lw=0, cmap="pink")
		plt.colorbar(fraction=0.02, pad=0.04)
		selected = uf < 0.1
		plt.scatter(xy[selected, 0], xy[selected, 1], fc="none", ec="crimson", s=marker_size * 2, lw=0.5)
		plt.axis("off")
		plt.title("Unspliced fraction (%, red outline <10%)")

		# Mito
		plt.subplot(3, 2, 6)
		plt.scatter(xy[:, 0], xy[:, 1], c=self.MitoFraction[:] * 100, s=marker_size, lw=0, cmap="cividis")
		plt.axis("off")
		plt.title("Mitochondrial fraction (%)")
		plt.colorbar(fraction=0.02, pad=0.04)

		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
		plt.close()
