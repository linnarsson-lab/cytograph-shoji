from cytograph import Algorithm, requires
from .scatter import scatterc, scattern
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from typing import List

import cytograph.visualization as cgplot
import scipy.cluster.hierarchy as hc
from matplotlib.colors import ListedColormap


class PlotHmmKaryotype:
    def __init__(self, markers: List[str] = ["PTPRC", "MOG", "PLP1", "PDGFRA",  "CLDN5", "SOX2-OT", "PCNA", "DCN", "CD44", "SOX10", "AQP4"], filename: str = "karyotype.png", **kwargs):
        super().__init__(**kwargs)
        self.markers = markers
        self.filename = filename

    def fit(self, ws, save = False):
        gene_names = ws.Gene[:]
        xy = ws.KaryotypeEmbedding[:]
        plt.figure(figsize=(15, 15))
        plt.subplot(6, 6, 1)
        cgplot.scatterc(xy, s=5, c=ws.KaryotypeMetacells[:])
        plt.title("Metacells")
        plt.axis("off")
        cell_ordering = ws.KaryotypeCellOrdering[:]
        for i, gene in enumerate(self.markers):
            plt.subplot(6, 6, i + 2)
            x = ws.Expression[:, gene_names == gene].flatten()[cell_ordering]
            cgplot.scattern(xy, s=5, c=x, bgval=0, max_percentile=100)
            plt.axis("off")
            plt.title(gene)

        # make a colormap for the ploidy heatmaps
        bottom = plt.cm.get_cmap('Oranges', 128)
        middle = np.full((128, 4), 0.97)
        middle[:, 3] = 1
        top = plt.cm.get_cmap('Blues_r', 128)
        newcolors = np.vstack((top(np.linspace(0, 1, 128)), middle, bottom(np.linspace(0, 1, 128))))
        cmp = ListedColormap(newcolors, name='OrangeBlue')

        chromosome_borders = ws.KaryotypeChromosomeBorders[:]
        ploidy = ws.KaryotypePloidy[:]
        predicted_ploidy = ws.KaryotypePredictedPloidy[:]
        median_ploidy = ws.KaryotypeMedianPloidy[:]
        labels = ws.KaryotypeMetacells[:]
        dendrogram = ws.KaryotypeDendrogram[:]

        n_cells = ploidy.shape[0]
        n_metacells, n_windows = median_ploidy.shape

        plt.subplot(312)
        plt.imshow(ploidy, vmin=0, vmax=4, aspect="auto", cmap=cmp, interpolation="none", origin="upper", extent=(0, n_windows, 0, n_cells))
        offset = 0
        for lbl in np.unique(labels):
            j = (labels == lbl).sum()
            plt.hlines(n_cells - (j + offset), 0, n_windows, color="black", lw=0.5, ls="-")
            offset += j

        for x in chromosome_borders:
            plt.vlines(x, 0, n_cells, color="black", lw=0.5, ls="--")
       # plt.colorbar()
        
        plt.axis("off")
        plt.subplot(313)
        plt.imshow(predicted_ploidy, vmin=0, vmax=4, aspect="auto", cmap=cmp, interpolation="none", origin="upper", extent=(0, n_windows, 0, n_metacells))
        for x in chromosome_borders:
            plt.vlines(x, 0, n_metacells, color="black", lw=0.5, ls="--")

        i = 0
        for ch, j in enumerate(chromosome_borders):
            plt.text(i + (j - i)/2, 0.5, str(ch + 1) if ch <= 21 else "XY"[ch - 22], ha="center")
            i = j


        tumor_vs_normal_cutoff = n_metacells - (np.where(hc.cut_tree(dendrogram, n_clusters=2) == 0)[0].max() + 1)
        plt.hlines(tumor_vs_normal_cutoff, 0, n_windows, color="crimson", lw=0.5, ls="--")
        plt.axis("off")

        if save:
            plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
            plt.close()
