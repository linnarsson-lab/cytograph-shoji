import cytograph as cg
import cytograph.visualization as cgplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from numpy_groupies.aggregate_numpy import aggregate
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.cluster.hierarchy import cut_tree
from cytograph import Algorithm, requires
import logging
import shoji


def sparkline(ax, x, ymax, color, plot_label, labels, subtrees):
	n_clusters = labels.max() + 1
	if ymax is None:
		ymax = np.max(x)
	ax.bar(np.arange(n_clusters) + 0.5, x, color=color, width=1, lw=0)
	ax.set_xlim(0, n_clusters + 1)
	ax.set_ylim(0, ymax)
	ax.axis("off")
	ax.text(0, 0, plot_label, va="bottom", ha="right", transform=ax.transAxes)
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 1, 0, ymax, linestyles="--", lw=1, color="grey")


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
	ax.imshow(color, cmap=plt.cm.Reds, aspect='auto', interpolation="none", origin="upper", extent=(0, n_clusters, n_classes, 0))
	ax.set_yticks(np.arange(n_classes) + 0.5)
	ax.set_yticklabels(classes)
	ax.set_xticks([])
	ax.hlines(n_classes, 0, n_clusters, lw=1, linestyles="--", color="grey")
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 1, 0, n_classes, linestyles="--", lw=1, color="grey")
	ax.set_ylim(n_classes, 0)
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

	ax.imshow(distro, cmap=plt.cm.Reds, aspect='auto', interpolation="none", origin="lower", extent=(0, n_clusters, 0, n_classes))
	ax.set_yticks(np.arange(len(classes)) + 0.5)
	ax.set_yticklabels(classes)
	ax.hlines(np.arange(5, len(classes), 5), 0, n_clusters, lw=1, linestyles="--", color="grey")
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 1, 0, len(classes), linestyles="--", lw=1, color="grey")
	ax.set_ylim(len(classes), 0)
	ax.set_xlim(0, n_clusters)
	ax.set_ylabel("Age (p.c.w.)")
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)


def plot_genes(ax, markers, mean_x, genes, labels, subtrees, enriched_genes):
	# Add the markers
	m = []
	m_names = []
	for category, mgenes in markers.items():
		for gene in mgenes:
			try:
				gene_ix = np.where(genes == gene)[0][0]
				m.append(mean_x[:, gene_ix])
				m_names.append(f"{gene} ({category})")
			except:
				pass
	n_genes = len(m_names)
	n_clusters = labels.max() + 1

	# Normalize
	x = np.array(m)
	totals = mean_x.sum(axis=1)
	x_norm = (x / totals * np.median(totals))

	ax.imshow(np.log10(x_norm + 0.001), vmin=-1, vmax=2, cmap="RdGy_r", interpolation="none", aspect="auto", extent=(0, n_clusters, n_genes, 0))
	ax.set_yticks(np.arange(len(m_names)) + 0.5)
	ax.set_yticklabels(m_names)
	if n_clusters < 200:
		ax.set_xticks(np.arange(n_clusters) + 0.5)
		ax.set_xticklabels(enriched_genes, rotation=60, rotation_mode='anchor', fontsize=min(12, 20 / n_clusters * 72 * 0.6), verticalalignment='top', horizontalalignment='right')
	else:
		subtree_right_edges = np.array([0] + [labels[subtrees == ix].max() for ix in range(subtrees.max() + 1)])
		subtree_centers = subtree_right_edges[:-1] + np.diff(subtree_right_edges) / 2
		ax.set_xticks(subtree_centers + 1)
		ax.set_xticklabels(np.arange(subtrees.max() + 1), rotation=60, rotation_mode='anchor', fontsize=min(12, 20 / n_clusters * 72 * 0.6), verticalalignment='top', horizontalalignment='right')
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 1, 0, n_genes, linestyles="--", lw=1, color="grey")

	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)


def plot_auto_annotation(ax, ann_names, ann_post, labels, subtrees):
	n_anns = len(ann_names)
	n_clusters = labels.max() + 1

	ax.imshow(ann_post, cmap=plt.cm.Purples, vmin=0, vmax=1, aspect='auto', interpolation="none", origin="upper", extent=(0, n_clusters, n_anns, 0))
	ax.set_yticks(np.arange(n_anns) + 0.5)
	ax.set_yticklabels(ann_names)
	ax.hlines(np.arange(5, n_anns, 5), 0, n_clusters, lw=1, linestyles="--", color="grey")
	for ix in range(subtrees.max() + 1):
		ax.vlines(labels[subtrees == ix].max() + 1, 0, n_anns, linestyles="--", lw=1, color="grey")
	ax.set_ylim(len(ann_names), 0)

	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)


def plot_dendrogram(ax, n_clusters, linkage, labels, subtrees):
	lc = cgplot.dendrogram(linkage, leaf_positions=np.arange(n_clusters) + 0.5)
	ax.add_collection(lc)
	ax.set_xlim(0, n_clusters)
	ax.set_ylim(0, linkage[:, 2].max())
	if n_clusters < 100:
		for i in range(n_clusters):
			ax.text(i + 0.5, 0, str(i), rotation=60, rotation_mode='anchor', fontsize=min(12, 20 / n_clusters * 72 * 0.6), verticalalignment='top', horizontalalignment='right')
	else:
		for ix in range(subtrees.max() + 1):
			ax.text(labels[subtrees == ix].max() + 0.5, 0, str(labels[subtrees == ix].max()), rotation=60, rotation_mode='anchor', fontsize=min(12, 20 / n_clusters * 72 * 0.6), verticalalignment='top', horizontalalignment='right')
	ax.axis("off")


class PlotOverview(Algorithm):
	def __init__(self, filename: str = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename if filename is not None else "_overview.png"

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
	@requires("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotOverview: Plotting the heatmap")

		labels = self.Clusters[:]
		regions = self.Region[:]
		subregions = self.Subregion[:]
		ages = self.Age[:].astype(int)
		n_clusters = ws.clusters.length

		cluster_labels = self.ClusterID[:]
		ordering = np.argsort(cluster_labels)
		mean_x = self.MeanExpression[:][ordering]
		markers = cg.Species(self.Species[:]).markers
		genes = self.Gene[:]
		enrichment = self.Enrichment[:]
		enriched_genes = []
		for i in range(ws.clusters.length):
			enriched_genes.append(" ".join(genes[np.argsort(-enrichment[cluster_labels == i, :][0])[:5]]))

		ordering = indices_to_order_a_like_b(self.ClusterID[:], np.arange(n_clusters))
		ann_names = self.AnnotationName[:]
		ann_post = self.AnnotationPosterior[:].T[:, ordering]
		n_cells = self.NCells[:]
		linkage = self.Linkage[:].astype("float64")

		if "Subtree" in ws:
			subtrees = ws.Subtree[:]
		else:
			# Cut the tree into subtrees with about ten clusters each
			cluster_subtrees = cut_tree(linkage, n_clusters=max(1, n_clusters // 10)).flatten().astype("int")
			# Map this onto the individual cells
			subtrees = np.zeros(ws.cells.length, dtype="uint32")
			for j in range(n_clusters):
				subtrees[labels == j] = cluster_subtrees[j]

		fig, axes = plt.subplots(nrows=10, ncols=1, sharex=True, gridspec_kw={"height_ratios": (2, 0.25, 0.25, 0.25, 0.25, 2, 2, 6, 8, 12)}, figsize=(20, 33))
		plot_dendrogram(axes[0], n_clusters, linkage, labels, subtrees)
		sparkline(axes[1], n_cells, None, "orange", "Cells", labels, subtrees)
		sparkline(axes[2], aggregate(labels, self.TotalUMIs[:], func="mean"), None, "green", "TotalUMIs", labels, subtrees)
		sparkline(axes[3], aggregate(labels, self.CellCycleFraction[:], func="mean"), 0.05, "blue", "Cell cycle", labels, subtrees)

		sparkline(axes[4], aggregate(labels, self.DoubletScore[:], func="mean"), 0.4, "crimson", "Doublet score", labels, subtrees)
		plot_ages(axes[5], ages, labels, subtrees)
		plot_regions(axes[6], regions, cgplot.Colorizer("regions").dict(), labels, subtrees)
		plot_regions(axes[7], subregions, cgplot.Colorizer("subregions").dict(), labels, subtrees)
		plot_auto_annotation(axes[8], ann_names, ann_post, labels, subtrees)
		plot_genes(axes[9], markers, mean_x, genes, labels, subtrees, enriched_genes)
		fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		if save:
			plt.savefig(self.export_dir / (ws._name + self.filename), dpi=300, bbox_inches='tight')
			plt.close()

class PlotOverviewEEL(Algorithm):
	def __init__(self, filename: str = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename if filename is not None else "_overview.png"

	@requires("Species", "string", ())
	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("CellCycleFraction", "float32", ("cells",))
	@requires("MeanExpression", "float64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("AnnotationName", "string", ("annotations",))
	@requires("AnnotationPosterior", "float32", ("clusters", "annotations"))
	@requires("NCells", "uint64", ("clusters",))
	@requires("Linkage", "float32", None)
	@requires("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotOverview: Plotting the heatmap")

		labels = self.Clusters[:]
		n_clusters = ws.clusters.length

		cluster_labels = self.ClusterID[:]
		ordering = np.argsort(cluster_labels)
		mean_x = self.MeanExpression[:][ordering]
		markers = cg.Species(self.Species[:]).gb_markers
		genes = self.Gene[:]
		enrichment = self.Enrichment[:]
		enriched_genes = []
		for i in range(ws.clusters.length):
			enriched_genes.append(" ".join(genes[np.argsort(-enrichment[cluster_labels == i, :][0])[:5]]))

		ordering = indices_to_order_a_like_b(self.ClusterID[:], np.arange(n_clusters))
		ann_names = self.AnnotationName[:]
		ann_post = self.AnnotationPosterior[:].T[:, ordering]
		n_cells = self.NCells[:]
		linkage = self.Linkage[:].astype("float64")

		if "Subtree" in ws:
			subtrees = ws.Subtree[:]
		else:
			# Cut the tree into subtrees with about ten clusters each
			cluster_subtrees = cut_tree(linkage, n_clusters=max(1, n_clusters // 10)).flatten().astype("int")
			# Map this onto the individual cells
			subtrees = np.zeros(ws.cells.length, dtype="uint32")
			for j in range(n_clusters):
				subtrees[labels == j] = cluster_subtrees[j]

		fig, axes = plt.subplots(nrows=10, ncols=1, sharex=True, gridspec_kw={"height_ratios": (2, 0.25, 0.25, 0.25, 0.25, 2, 2, 6, 8, 12)}, figsize=(20, 33))
		plot_dendrogram(axes[0], n_clusters, linkage, labels, subtrees)
		sparkline(axes[1], n_cells, None, "orange", "Cells", labels, subtrees)
		sparkline(axes[2], aggregate(labels, self.TotalUMIs[:], func="mean"), None, "green", "TotalUMIs", labels, subtrees)
		sparkline(axes[3], aggregate(labels, self.CellCycleFraction[:], func="mean"), 0.05, "blue", "Cell cycle", labels, subtrees)
		#sparkline(axes[4], aggregate(labels, self.DoubletScore[:], func="mean"), 0.4, "crimson", "Doublet score", labels, subtrees)
		#plot_ages(axes[5], ages, labels, subtrees)
		#plot_regions(axes[6], regions, cgplot.Colorizer("regions").dict(), labels, subtrees)
		#plot_regions(axes[7], subregions, cgplot.Colorizer("subregions").dict(), labels, subtrees)
		plot_auto_annotation(axes[8], ann_names, ann_post, labels, subtrees)
		plot_genes(axes[9], markers, mean_x, genes, labels, subtrees, enriched_genes)
		fig.tight_layout(pad=0, h_pad=0, w_pad=0)

		if save:
			plt.savefig(self.export_dir / (ws._name + self.filename), dpi=300, bbox_inches='tight')
			plt.close()
