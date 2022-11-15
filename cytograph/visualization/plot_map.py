from ensurepip import bootstrap
from typing import Dict, List
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shoji
from cytograph import Algorithm, requires
from colormap import rgb2hex
from .scatter import scatterc, scattern
from .colors import colorize
from colormap import rgb2hex
from scipy.spatial import KDTree
import holoviews as hv
hv.notebook_extension('bokeh')

def indices_to_order_a_like_b(a, b):
	return a.argsort()[b.argsort().argsort()]

#class SpatialMap(Algorithm):
class PlotSpatialmap(Algorithm):
    def __init__(self, filename='spatialmap',backend='holoviews', cmap=None, point_size=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.filename = filename
        self.backend = backend
        self.cmap = cmap
        self.point_size = point_size

    @requires("Gene", "string", ("genes",))
    @requires("ClusterID", "uint32", ("clusters",))
    @requires("Clusters", "uint32", ("cells",))
    @requires("GraphCluster", "int8", ("cells",))
    @requires("MeanExpression", "float64", ("clusters", "genes"))
    @requires("X", "float32", ("cells",))
    @requires("Y", "float32", ("cells",))
    @requires("Enrichment", "float32", ("clusters", "genes"))
    @requires("NCells", "uint64", ("clusters",))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        labels = []
        n_cells = self.NCells[:]
        clusters = self.Clusters[:]
        n_clusters = clusters.max() + 1
        cluster_ids = self.ClusterID[:]
        genes = self.Gene[:]
        enrichment = self.Enrichment[:]
        clusters_label_dic = {}

        for i in range(ws.clusters.length):
            n = n_cells[cluster_ids == i][0]
            label = f"{i:>3} ({n:,} cells) "
            label += " ".join(genes[np.argsort(-enrichment[cluster_ids == i, :][0])[:10]])
            labels.append(label)
            clusters_label_dic[i] = label
        labels = np.unique(labels)

        clusters = self.Clusters[:]
        x = self.X[:]
        y = self.Y[:]
        data = pd.DataFrame({'x': x, 'y':y, 'cluster':clusters}) #np.concatenate([xy,clusters[:,np.newaxis]],axis=1)
        #data.loc[:,'cluster'] = np.array([clusters_label_dic[c] for c in data['cluster']])

        unique_colors = colorize(np.unique(clusters))
        unique_colors_HEX = [rgb2hex(int(color[0]*255),int(color[1]*255),int(color[2]*255)) for color in unique_colors]
        if type(self.cmap) != type(None):
            cmap = self.cmap
        else:
            cmap = {t:col for t,col in zip(labels,unique_colors_HEX)}
        #logging.info('cmap',cmap)
        print(cmap)
        if self.backend == 'holoviews':
            dic = {}
            for cluster, d in data.groupby('cluster'):
                print(cluster)
                scatter = hv.Scatter(d,kdims=['x'],vdims=['y','cluster']).opts(
                                                                        color=cmap[clusters_label_dic[cluster]], 
                                                                        width=1000,
                                                                        size=self.point_size, 
                                                                        xticks=0,
                                                                        yticks=0, 
                                                                        ylabel=None,
                                                                        xlabel=None,
                                                                        xaxis=None,
                                                                        yaxis=None, 
                                                                        bgcolor='black',
                                                                        aspect='equal',
                                                                        )
                dic[clusters_label_dic[i]] = scatter
            scatter = hv.NdOverlay(dic).opts(show_legend=True,legend_limit=100,)
        elif self.backend == 'matplotlib':
            dic = {}
            for cluster, d in data.groupby('cluster'):
                print(cluster)
                scatter = hv.Scatter(d,kdims=['x'],vdims=['y','cluster']).opts(
                                                                        color=cmap[cluster], 
                                                                        width=800, 
                                                                        height=800,
                                                                        s=self.point_size, 
                                                                        xticks=0,
                                                                        yticks=0, 
                                                                        ylabel=None,
                                                                        xlabel=None,
                                                                        xaxis=None,
                                                                        yaxis=None, 
                                                                        )
                dic[cluster] = scatter
            scatter = hv.NdOverlay(dic).opts(show_legend=True)

        if save:
            hv.save(scatter, self.export_dir / (ws._name + "_map.png"),dpi=500)
            hv.save(scatter, self.export_dir / (ws._name + "_map.html"),dpi=500)

        return scatter