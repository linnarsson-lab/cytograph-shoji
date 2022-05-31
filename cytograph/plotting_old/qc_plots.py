import matplotlib.pyplot as plt
import numpy as np
import loompy
from .colors import colorize
from cytograph.enrichment import FeatureSelectionByVariance
from cytograph.embedding import art_of_tsne
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def all_QC_plots(ds: loompy.LoomConnection = None, out_file: str = "Tmp.png"):
	f, ax = plt.subplots(4, 2, figsize=(2 * 8, 4 * 8))
	
	plot_title = "Mitochondrial gene expression ratio distribution"
	dist_attr(ax[0, 0], ds, attr = "MT_ratio", plot_title= plot_title, line=0.05)
	plot_title = "Unspliced reads ratio distribution"
	dist_attr(ax[0, 1], ds, attr = "unspliced_ratio", plot_title= plot_title, line=0.5)
	attrs_on_TSNE(ax[1:4, 0:2], ds, attrs = ["DoubletFinderFlag", "DoubletFinderScore", "TotalUMI", "NGenes", "unspliced_ratio", "MT_ratio"], plot_title = ["Doublets Flag", "Doublets Score", "UMI counts", "Number of genes per cell", "Unspliced UMI ratio", "Mitochondrial genes ratio"])
	
	f.savefig(out_file, dpi=144)
	plt.close(f)


def dist_attr(ax: plt.axes = None, ds: loompy.LoomConnection = None, out_file: str = None, attr: str = None, plot_title: str = None, line: float = None):

	if ax is None:
		fig, ax = plt.subplots(figsize=(6, 4))
	if attr in ds.ca:
		ax.hist(ds.ca[attr], bins = 100)
		ax.set_xlim(np.amin(ds.ca[attr]), np.amax(ds.ca[attr]))
		if line is not None:
			ax.axvline(x=line, c='r')
		if plot_title is not None:
			ax.set_title(plot_title)
		if out_file is not None:
			fig.savefig(out_file, dpi=144)
		return(ax)


def attrs_on_TSNE(ax: plt.axes = None, ds: loompy.LoomConnection = None, out_file: str = None, attrs: list = None, plot_title: list = None):
	n_attr = len(attrs)
	n_cols = 2
	n_rows = np.ceil(n_attr / n_cols)
	n_rows = n_rows.astype(int)
	if ax is None and n_attr > 1:
		fig, ax = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 8 * n_rows))
	elif ax is None and n_attr == 1:
		fig, ax = plt.subplots(figsize=(12, 12))
	n_cols_plot = 0
	n_rows_plot = 0
	if 'TSNE' in ds.ca:
		xy = ds.ca.TSNE
	elif 'HPF' in ds.ca:
		xy = art_of_tsne(ds.ca.HPF)
		ds.ca.TSNE = xy
	elif 'PCA' in ds.ca:
		angle = 0.5
		perplexity = 30
		verbose = False
		xy = TSNE(angle=angle, perplexity=perplexity, verbose=verbose).fit_transform(ds.ca.PCA)
		ds.ca.TSNE = xy
	else:

		genes = FeatureSelectionByVariance(2000).fit(ds)
		data = ds[:, :]
		f = np.divide(data.sum(axis=0), 10e6)
		norm_data = np.divide(data, f)
		norm_data = np.log(norm_data + 1)
		ds.ca.PCA = PCA(n_components=50).fit_transform(norm_data[genes, :].T)
		angle = 0.5
		perplexity = 30
		verbose = False
		xy = TSNE(angle=angle, perplexity=perplexity, verbose=verbose).fit_transform(ds.ca.PCA)
		ds.ca.TSNE = xy
	
	for n, attr in enumerate(attrs):
		# ax.append( plt.subplot(n_rows,n_cols,n+1))
		if(n_attr == 1):
			current_ax = ax
		if(n_attr == 2):
			current_ax = ax[n_cols_plot]
		if(n_attr > 2):
			current_ax = ax[n_rows_plot, n_cols_plot]
		if attr in ds.ca:
			current_ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', lw=0, s=10)
			names, labels = np.unique(ds.ca[attr], return_inverse=True)
			if (len(names) > 20):
				labels = ds.ca[attr]
				if(np.max(ds.ca[attr]) > 10000):
					labels = np.log(labels)
					if plot_title is not None:
						current_ax.set_title(plot_title[n] + " (log transformed)")
				elif plot_title is not None:
					current_ax.set_title(plot_title[n])
				cm = plt.cm.get_cmap('jet')
				cells = ds.ca[attr] > 0
				sc = current_ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=labels[cells], lw=0, s=10, cmap = cm)
				plt.colorbar(sc, ax=current_ax)
			else:
				colors = colorize(names)
				cells = np.random.permutation(labels.shape[0])
				current_ax.scatter(xy[cells, 0], xy[cells, 1], c=colors[labels][cells], lw=0, s=10)

				def h(c):
					return plt.Line2D([], [], color=c, ls="", marker="o")
				current_ax.legend(handles=[h(colors[i]) for i in range(len(names))], labels=list(names), loc='lower left', markerscale=1, frameon=False, fontsize=10)
				if plot_title is not None:
					current_ax.set_title(plot_title[n])
		else:
			current_ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', lw=0, s=10)
			if plot_title is not None:
				current_ax.set_title(plot_title[n])
		if(n_cols_plot != n_cols - 1):
			n_cols_plot = n_cols_plot + 1
		else:
			n_cols_plot = 0
			n_rows_plot = n_rows_plot + 1
	if out_file is not None:
		fig.savefig(out_file, dpi=144)
	return(ax)
