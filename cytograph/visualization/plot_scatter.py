import logging

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import shoji
from cytograph import Algorithm, requires
from .scatter import scatterc, scattern


class NumericalScatterplot(Algorithm):
	def __init__(self, tensor: str, width: float = 8, height: float = 8, cmap: str = "viridis", vmin: float = None, vmax: float = None, vcenter: float = None, filename: str = "scatter.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.tensor = tensor
		self.width = width
		self.height = height
		self.cmap = cmap
		self.vmin = vmin
		self.vmax = vmax
		self.vcenter = vcenter
		self.filename = filename

	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" NumericalScatterplot: Loading the embedding")
		xy = self.Embedding[:]

		logging.info(f" NumericalScatterplot: Plotting '{self.tensor}'")
		n_cells = ws.cells.length
		plt.figure(figsize=(self.width, self.height))

		def expand_scalars(x):
			if x.ndim == 0:
				return np.full(n_cells, fill_value=x)
			else:
				return x

		data = expand_scalars(ws[self.tensor][:])
		if self.vcenter is not None:
			scattern(xy, c=data, cmap=self.cmap, norm=colors.CenteredNorm(self.vcenter))
		else:
			scattern(xy, c=data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
		plt.colorbar(fraction=0.02, pad=0.04)
		plt.axis("off")
		plt.title(self.tensor)

		try:
			plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
			plt.close()
		except AttributeError:
			pass


class CategoricalScatterplot(Algorithm):
	def __init__(self, tensor: str, width: float = 8, height: float = 8, legend: str = "outside", filename: str = "scatter.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.tensor = tensor
		self.width = width
		self.height = height
		self.legend = legend
		self.filename = filename

	@requires("Embedding", "float32", ("cells", 2))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" CategoricalScatterplot: Loading the embedding")
		xy = self.Embedding[:]

		logging.info(f" CategoricalScatterplot: Plotting '{self.tensor}'")
		n_cells = ws.cells.length
		plt.figure(figsize=(self.width, self.height))

		def expand_scalars(x):
			if x.ndim == 0:
				return np.full(n_cells, fill_value=x)
			else:
				return x

		data = expand_scalars(ws[self.tensor][:])
		scatterc(xy, c=data, legend=self.legend)
		plt.axis("off")
		plt.title(self.tensor)

		try:
			plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
			plt.close()
		except AttributeError:
			pass
