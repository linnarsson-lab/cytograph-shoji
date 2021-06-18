import matplotlib.pyplot as plt
import numpy as np
import shoji
from cytograph import Module, Species, requires
from .dendrogram import dendrogram
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Heatmap(Module):
	def __init__(self, filename: str = "heatmap.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename
	
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@requires("Gene", "string", ("genes",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	@requires("Species", "string", ())
	@requires("MeanAge", "float64", ("clusters",))
	@requires("NCells", "uint64", ("clusters",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False):
		# Load data
		genes = self.Gene[:]
		ordering = np.argsort(ws.ClusterID[:])
		mean_x = self.MeanExpression[:][ordering]
		enrichment = self.Enrichment[:][ordering]
		markers = Species(self.Species[:]).markers

		# Compute the main heatmap
		enriched_genes = []
		x = []
		for i in range(enrichment.shape[0]):
			count = 0
			for gene in genes[np.argsort(-enrichment[i, :])]:
				gene_ix = np.where(genes == gene)[0][0]
				if gene not in enriched_genes:
					enriched_genes.append(gene)
					x.append(mean_x[:, gene_ix])
					count += 1
				if count == 3:
					break
		x = np.array(x)  # x is (n_genes, n_clusters)
		enriched_genes = np.array(enriched_genes)
		# Rearrange the genes by the max-expressing cluster
		top_cluster = []
		for g in enriched_genes:
			top_cluster.append(np.argsort(-enrichment[:, genes == g].T[0])[0])
		gene_ordering = np.argsort(top_cluster)
		x = x[gene_ordering, :]
		enriched_genes = enriched_genes[gene_ordering]
		# Add the markers
		m = []
		m_names = []
		for category, mgenes in markers.items():
			for gene in mgenes:
				gene_ix = np.where(genes == gene)[0][0]
				m.append(mean_x[:, gene_ix])
				m_names.append(gene)
		x = np.vstack([m, x])
		enriched_genes = np.concatenate([m_names, enriched_genes])
		# Normalize
		totals = mean_x.sum(axis=1)
		x_norm = (x / totals * np.median(totals)).T
		#x_norm = x_norm / np.max(x_norm, axis=0)
		#x_norm = x.T
		
		# Set up the figure
		n_genes = x.shape[0]
		n_clusters = ws.clusters.length
		dendrogram_height = n_genes / 40
		fig_width = n_clusters / 40
		fig_height = dendrogram_height + (1 + 2 + n_genes) / 40
		fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)
		
		heights = [dendrogram_height] + [1, 2, n_genes]
		fig_spec = fig.add_gridspec(nrows=4, height_ratios=heights)
		fig_spec.hspace = 0
		fig_spec.wspace = 0
		subplot = 0
		
		# Plot the dendrogram
		ax = fig.add_subplot(fig_spec[subplot])
		z = ws.Linkage[:].astype("float64")
		lines = dendrogram(z)
		lines.set_linewidth(0.5)
		ax.add_collection(lines)
		ax.set_xlim(-0.5, ws.clusters.length - 0.5)
		ax.set_ylim(0, z[:, 2].max() * 1.1)
		plt.axis("off")
		# Plot the legend for the main heatmap
		inset = inset_axes(ax, width="10%", height="20%", borderpad=0.1, loc=2)
		inset.imshow(np.tile(np.arange(-1, 2, 0.1), (10, 1)), vmin=-1, vmax=2, cmap="RdGy_r", interpolation="none", aspect="equal")
		for axis in ['top', 'bottom', 'left', 'right']:
			inset.spines[axis].set_linewidth(0.5)
		inset.tick_params(length=1, width=0.5)
		inset.set_xticks([0, 10, 20, 30])
		inset.set_xticklabels(["0.1", "1", "10", "100"], fontsize=2, rotation=90)
		inset.set_yticks([])
		subplot += 1

		# Plot the stripe of ages
		ax = fig.add_subplot(fig_spec[subplot])
		if "MeanAge" in ws:
			ages = ws.MeanAge[:][ordering]
			ax.imshow(ages[None, :], cmap="rainbow", vmin=min(ages[ages > 0]), vmax=max(ages))
			ax.set_xlim(-0.5, n_clusters - 0.5)
			plt.axis("off")
		subplot += 1

		# Plot the barchart of cluster sizes
		n_cells = ws.NCells[:][ordering]
		ax = fig.add_subplot(fig_spec[subplot])
		ax.bar(np.arange(n_clusters), n_cells, color="grey")
		ax.set_xlim(-0.5, n_clusters - 0.5)
		plt.axis("off")
		subplot += 1

		# Plot the heatmap
		ax = fig.add_subplot(fig_spec[subplot])
		ax.set_anchor("N")
		ax.imshow(np.log10(x_norm.T + 0.001), vmin=-1, vmax=2, cmap="RdGy_r", interpolation="none", aspect="equal")
		ax.set_yticks(np.arange(x_norm.shape[1]))
		ax.set_yticklabels(enriched_genes, fontsize=2)
		ax.set_xticks(np.arange(0, x_norm.shape[0], 10) - 0.5)
		ax.set_xticklabels(np.arange(0, x_norm.shape[0], 10), fontsize=2)
		for ix, label in enumerate(enriched_genes):
			if ix < len(enriched_genes) - 21:
				ax.text(np.argmax(x_norm[:, ix]) + 0.7, ix + 0.3, label, fontsize=2, color="white")
			else:
				ax.text(np.argmax(x_norm[:, ix]) - 0.5, ix + 0.3, label, fontsize=2, color="white", ha="right")
		for axis in ['top', 'bottom', 'left', 'right']:
			ax.spines[axis].set_linewidth(0.5)
		ax.tick_params(width=0.5)

		plt.tight_layout()
		plt.subplots_adjust(hspace=0, left=0, right=1, top=1, bottom=0)
		
		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=400, bbox_inches='tight')
		plt.close()
