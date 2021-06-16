from typing import Optional
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from .colors import colorize


def _draw_edges(ax: plt.Axes, pos: np.ndarray, g: np.ndarray, gcolor: str, galpha: float, glinewidths: float) -> None:
	lc = LineCollection(zip(pos[g.row], pos[g.col]), linewidths=0.25, zorder=0, color=gcolor, alpha=galpha)
	ax.add_collection(lc)


def scatterc(xy: np.ndarray, *, c: np.ndarray, legend: Optional[str] = "outside", g: np.ndarray = None, gcolor: str = "thistle", galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
	n_cells = xy.shape[0]
	marker_size = 500_000 / n_cells

	ordering = np.random.permutation(xy.shape[0])
	c = c[ordering]
	xy = xy[ordering, :]
	s = kwargs.pop("s", marker_size)
	lw = kwargs.pop("lw", 0)
	plt.scatter(xy[:, 0], xy[:, 1], c=colorize(c), s=s, lw=lw, **kwargs)
	ax = plt.gca()
	if legend not in [None, False]:
		hidden_lines = [Line2D([0], [0], color=clr, lw=4) for clr in colorize(np.unique(c))]
		if legend == "outside":
			ax.legend(hidden_lines, np.unique(c), loc='center left', bbox_to_anchor=(1, 0.5))
		else:
			ax.legend(hidden_lines, np.unique(c), loc=legend)
	if g is not None:
		_draw_edges(ax, xy, g, gcolor, galpha, glinewidths)
	plt.title(f"{n_cells:,} cells")

def scattern(xy: np.ndarray, *, c: np.ndarray, zinf: bool = True, g: np.ndarray = None, gcolor: str = "thistle", galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
	n_cells = xy.shape[0]
	marker_size = 1000 / np.sqrt(n_cells)

	ordering = np.random.permutation(xy.shape[0])
	color = c[ordering]
	xy = xy[ordering, :]
	s = kwargs.pop("s", marker_size)
	lw = kwargs.pop("lw", 0)
	if zinf:
		cells = color > 0
		plt.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=s, lw=lw, **kwargs)
		plt.scatter(xy[cells, 0], xy[cells, 1], c=color[cells], s=s, lw=lw, **kwargs)
	else:
		plt.scatter(xy[:, 0], xy[:, 1], c=color, s=s, lw=lw, **kwargs)
	if g is not None:
		ax = plt.gca()
		_draw_edges(ax, xy, g, gcolor, galpha, glinewidths)

