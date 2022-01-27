import cytograph as cg
import cytograph.visualization as cgplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from numpy_groupies.aggregate_numpy import aggregate
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from cytograph import Module, requires
import logging
import shoji


def sparkline(ax, x, ymax, color, plot_label, labels, subtrees):
	n_clusters = labels.max() + 1
	if ymax is None:
		ymax = np.max(x)
	ax.bar(np.arange(n_clusters), x, color=color, width=1, lw=0)
	ax.set_xlim(0, n_clusters)
	ax.set_ylim(0, ymax)
	ax.axis("off")
	ax.text(0, 0, plot_label, va="bottom", ha="right", transform=ax.transAxes)
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 0.5, 0, ymax, linestyles="--", lw=1, color="grey")


def indices_to_order_a_like_b(a, b):
	return a.argsort()[b.argsort().argsort()]


def plot_regions(ax, regions, region_colors, labels, subtrees):
	classes = np.array(list(region_colors.keys()))
	n_classes = len(classes)
	le = OrdinalEncoder(categories=[classes])
	le.fit(regions.reshape(-1, 1))
	n_clusters = labels.max() + 1
	distro = np.zeros((n_classes, n_clusters))
	for label in np.arange(n_clusters):
		subset = le.transform(regions[labels == label].reshape(-1, 1)).flatten().astype("int32")
		d = aggregate(subset, subset, func="count", size=n_classes)
		distro[:, label] = d

	opacity = distro / distro.sum(axis=0)
	color = np.zeros((n_classes, n_clusters, 4))
	color[:, :, 3] = opacity
	for ix, cls in enumerate(classes):
		color[ix, :, :3] = to_rgb(region_colors[cls])
	ax.imshow(color, cmap=plt.cm.Reds, aspect='auto', interpolation="none", origin="upper")
	ax.set_yticks(np.arange(n_classes))
	ax.set_yticklabels(classes)
	ax.set_xticks([])
	ax.hlines(n_classes - 0.5, 0, n_clusters - 0.5, lw=1, linestyles="--", color="grey")
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 0.5, -0.5, n_classes - 0.5, linestyles="--", lw=1, color="grey")
	ax.set_ylim(n_classes - 0.5, -0.5)
	ax.set_xlim(0, n_clusters)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)


def plot_ages(ax, ages, labels, subtrees):
	le = LabelEncoder()
	le.fit(ages)
	n_clusters = labels.max() + 1
	n_classes = len(le.classes_)
	distro = np.zeros((n_classes, n_clusters))
	for label in np.arange(n_clusters):
		subset = le.transform(ages[labels == label])
		d = aggregate(subset, subset, func="count", size=n_classes)
		distro[:, label] = d

	distro = (distro.T / distro.sum(axis=1)).T
	distro = distro / distro.sum(axis=0)
	classes = le.classes_

	ax.imshow(distro, cmap=plt.cm.Reds, aspect='auto', interpolation="none", origin="lower")
	ax.set_yticks(np.arange(len(classes)))
	ax.set_yticklabels(classes)
	ax.hlines(np.arange(4.5, len(classes), 5), 0, n_clusters - 0.5, lw=1, linestyles="--", color="grey")
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 0.5, -0.5, len(classes) - 0.5, linestyles="--", lw=1, color="grey")
	ax.set_ylim(len(classes) - 0.5, -0.5)
	ax.set_xlim(0, n_clusters)
	ax.set_ylabel("Age (p.c.w.)")
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)


def plot_genes(ax, markers, mean_x, genes, labels, subtrees):
	# Add the markers
	m = []
	m_names = []
	for category, mgenes in markers.items():
		for gene in mgenes:
			gene_ix = np.where(genes == gene)[0][0]
			m.append(mean_x[:, gene_ix])
			m_names.append(f"{gene} ({category})")
	# Normalize
	x = np.array(m)
	totals = mean_x.sum(axis=1)
	x_norm = (x / totals * np.median(totals))

	ax.imshow(np.log10(x_norm + 0.001), vmin=-1, vmax=2, cmap="RdGy_r", interpolation="none", aspect="auto")
	ax.set_yticks(np.arange(len(m_names)))
	ax.set_yticklabels(m_names)
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 0.5, -0.5, len(m_names) - 0.5, linestyles="--", lw=1, color="grey")
	
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)


def plot_auto_annotation(ax, ann_names, ann_post, labels, subtrees):
	ax.imshow(ann_post, cmap=plt.cm.Purples, vmin=0, vmax=1, aspect='auto', interpolation="none", origin="upper")
	ax.set_yticks(np.arange(len(ann_names)))
	ax.set_yticklabels(ann_names)
	ax.hlines(np.arange(4.5, len(ann_names), 5), 0, ann_post.shape[1] - 0.5, lw=1, linestyles="--", color="grey")
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 0.5, -0.5, len(ann_names) - 0.5, linestyles="--", lw=1, color="grey")
	ax.set_ylim(len(ann_names) - 0.5, -0.5)
	
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)

	
def plot_dendrogram(ax, n_clusters, linkage):
	lc = cgplot.dendrogram(linkage)
	ax.add_collection(lc)
	ax.set_xlim(0, n_clusters)
	ax.set_ylim(0, linkage[:, 2].max())
	ax.axis("off")


class PlotOverview(Module):
	def __init__(self, filename: str = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename if filename is not None else "overview.png"

	@requires("Species", "string", ())
	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("CellCycleFraction", "float32", ("cells",))
	@requires("DoubletScore", "float32", ("cells",))
	@requires("Region", "string", ("cells",))
	@requires("Subregion", "string", ("cells",))
	@requires("Age", "float32", ("cells",))
	@requires("MeanExpression", "float64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("AnnotationName", "string", ("annotations",))
	@requires("AnnotationPosterior", "float32", ("clusters", "annotations"))
	@requires("NCells", "uint64", ("clusters",))
	@requires("Linkage", "float32", None)
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotOverview: Plotting the heatmap")
		if "Subtree" in ws:
			subtrees = self.Subtree[:]
		else:
			subtrees = np.zeros(ws.cells.length, dtype="int32")

		labels = self.Clusters[:]
		regions = self.Region[:]
		subregions = self.Subregion[:]
		ages = self.Age[:].astype(int)
		n_clusters = ws.clusters.length

		ordering = np.argsort(self.ClusterID[:])
		mean_x = self.MeanExpression[:][ordering]
		markers = cg.Species(self.Species[:]).markers
		genes = self.Gene[:]

		ordering = indices_to_order_a_like_b(self.ClusterID[:], np.arange(n_clusters))
		ann_names = self.AnnotationName[:]
		ann_post = self.AnnotationPosterior[:].T[:, ordering]
		n_cells = self.NCells[:]
		linkage = self.Linkage[:].astype("float64")

		fig, axes = plt.subplots(nrows=10, ncols=1, sharex=True, gridspec_kw={"height_ratios": (2, 0.25, 0.25, 0.25, 0.25, 2, 2, 6, 8, 12)}, figsize=(20, 33))
		plot_dendrogram(axes[0], n_clusters, linkage)
		sparkline(axes[1], n_cells, None, "orange", "Cells", labels, subtrees)
		sparkline(axes[2], aggregate(labels, self.TotalUMIs[:], func="mean"), None, "green", "TotalUMIs", labels, subtrees)
		sparkline(axes[3], aggregate(labels, self.CellCycleFraction[:], func="mean"), 0.05, "blue", "Cell cycle", labels, subtrees)
		sparkline(axes[4], aggregate(labels, self.DoubletScore[:], func="mean"), 0.4, "crimson", "Doublet score", labels, subtrees)
		plot_ages(axes[5], ages, labels, subtrees)
		plot_regions(axes[6], regions, {
			'Head': 'brown',
			'Brain': 'teal',
			'Forebrain': 'crimson',
			'Telencephalon': 'crimson',
			'Diencephalon': 'orange',
			'Midbrain': 'green',
			'Hindbrain': 'magenta',
			'Pons': 'magenta',
			'Cerebellum': 'magenta',
			'Medulla': 'blue'
		}, labels, subtrees)

		plot_regions(axes[7], subregions, {
			'Head': 'brown',
			'Brain': 'teal',

			'Forebrain': 'red',
			'Telencephalon': 'red',
			'Cortex': 'red',
			'Cortex hemisphere A': 'red',
			'Cortex hemisphere B': 'red',
			'Cortex lower': 'red',
			'Cortex upper': 'red',
			'Cortex occipital': 'red',
			'Cortex entorhinal': 'red',
			'Cortex frontotemporal': 'red',
			'Cortical hem': 'red',

			'Hippocampus': 'pink',

			'Subcortex': 'orange',
			'Caudate+Putamen': 'orange',
			'Striatum': 'orange',

			'Diencephalon': 'orange',
			'Hypothalamus': 'orange',
			'Thalamus': 'orange',

			'Midbrain': 'green',
			'Midbrain dorsal': 'green',
			'Midbrain ventral': 'green',

			'Hindbrain': 'magenta',
			'Cerebellum': 'magenta',
			'Pons': 'magenta',
			'Medulla': 'blue'
		}, labels, subtrees)
		plot_auto_annotation(axes[8], ann_names, ann_post, labels, subtrees)
		plot_genes(axes[9], markers, mean_x, genes, labels, subtrees)
		fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		if save:
			plt.savefig(self.export_dir / (ws._name + self.filename), dpi=300, bbox_inches='tight')
			plt.close()
