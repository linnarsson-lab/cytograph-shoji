from ensurepip import bootstrap
from typing import Dict, List
import logging
from os import path, makedirs
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
from bokeh.io import export_svgs
hv.notebook_extension('bokeh')

def indices_to_order_a_like_b(a, b):
	return a.argsort()[b.argsort().argsort()]


def export_webgl(obj, filename):
    plot_state = hv.renderer('bokeh').get_plot(obj).state
    plot_state.output_backend = 'webgl'
    export_svgs(plot_state, filename=filename)


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
    @requires("Sample", "string", ("cells",))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        samples = self.Sample[:]
        unique_samples = np.unique(self.Sample[:])
        n_cells = self.NCells[:]
        clusters = self.Clusters[:]
        n_clusters = clusters.max() + 1
        cluster_ids = self.ClusterID[:]
        genes = self.Gene[:]
        ordering = np.argsort(self.ClusterID[:])
        mean_x = self.MeanExpression[:][ordering]
        enrichment = self.Enrichment[:][ordering]
        idx_zeros = np.where(mean_x.sum(axis=0) == 0)[0]
        enrichment[:, idx_zeros] = np.zeros([enrichment.shape[0],idx_zeros.shape[0]])

        clusters_label_dic = {}
        labels = []
        for i in range(ws.clusters.length):
            n = n_cells[cluster_ids == i][0]
            label = f"{i:>3} ({n:,} cells) "
            label += " ".join(genes[np.argsort(-enrichment[cluster_ids == i, :][0])[:10]])
            labels.append(label)
            clusters_label_dic[i] = label
        labels = np.unique(labels)

        unique_colors = colorize(np.unique(clusters))
        unique_colors_HEX = [rgb2hex(int(color[0]*255),int(color[1]*255),int(color[2]*255)) for color in unique_colors]
        if type(self.cmap) != type(None):
            cmap = self.cmap
        else:
            cmap = {t:col for t,col in zip(labels,unique_colors_HEX)}

        for sample in unique_samples:
            filter_sample = samples == sample
            clusters_s = clusters[filter_sample]
            save_to = self.export_dir / 'Spatial_{}_{}'.format(ws._name , sample)
            if path.exists(save_to) == False:
                makedirs(save_to)

            x = self.X[:][filter_sample]
            y = self.Y[:][filter_sample]
            data = pd.DataFrame({'x': x, 'y':y, 'cluster':clusters_s}) #np.concatenate([xy,clusters[:,np.newaxis]],axis=1)
            #data.loc[:,'cluster'] = np.array([clusters_label_dic[c] for c in data['cluster']])

            if self.backend == 'holoviews':
                dicND = {}
                dicNDshortlabels = {}
                for cluster, d in data.groupby('cluster'):
                    #print(cluster)
                    scatter = hv.Scatter(d,kdims=['x'],vdims=['y','cluster']).opts(
                                                                            color=cmap[clusters_label_dic[cluster]], 
                                                                            height=1000,
                                                                            #width=1500,
                                                                            size=self.point_size, 
                                                                            xticks=0,
                                                                            yticks=0, 
                                                                            ylabel=None,
                                                                            xlabel=None,
                                                                            xaxis=None,
                                                                            yaxis=None, 
                                                                            bgcolor='black',
                                                                            aspect='equal',
                                                                            nonselection_fill_alpha=0,
                                                                            muted_fill_alpha=0,
                                                                            
                                                                            )
                    dicND[clusters_label_dic[cluster]] = scatter.opts(title=clusters_label_dic[cluster])
                    dicNDshortlabels[cluster] = scatter.opts(title=clusters_label_dic[cluster])

                ND = hv.NdOverlay(dicND).opts(
                    show_legend=True,legend_limit=100,legend_position='right',
                    fontsize={'legend':5},legend_spacing=-5,
                    )
                HM = hv.HoloMap(ND)
                
                NDshortlabels = hv.NdOverlay(dicNDshortlabels).opts(
                    show_legend=True,legend_limit=100,legend_position='right',
                    fontsize={'legend':5},legend_spacing=-5,
                    legend_muted=True,
                    )

                if save:
                    hv.save(HM, save_to / (ws._name + "_map_holomap.html"))
                    hv.save(NDshortlabels, save_to / (ws._name + "_map.html"))
                    NDshortlabels = NDshortlabels.opts(
                        hv.opts.Scatter(
                            size=0.05,
                        )
                    )
                    Layout = hv.Layout([dicNDshortlabels[x].opts(size=0.5, height=800,width=1600) for x in dicNDshortlabels]).cols(5)
                    hv.save(Layout, save_to / (ws._name + "_map.png"), dpi=2000)
                    
            elif self.backend == 'matplotlib':
                dic = {}
                for cluster, d in data.groupby('cluster'):
                    scatter = hv.Scatter(d,kdims=['x'],vdims=['y','cluster']).opts(
                                                                            color=cmap[cluster], 
                                                                            #width=800, 
                                                                            height=1000,
                                                                            s=self.point_size, 
                                                                            xticks=0,
                                                                            yticks=0, 
                                                                            ylabel=None,
                                                                            xlabel=None,
                                                                            xaxis=None,
                                                                            yaxis=None, 
                                                                            )
                    dic[clusters_label_dic[cluster]] = scatter
                ND = hv.NdOverlay(dic).opts(show_legend=True,legend_limit=100,nonselection_alpha=0)

                if save:
                    hv.save(ND, save_to / (ws._name + "_map.png"),dpi=500)
                    hv.save(ND, save_to / (ws._name + "_map.html"),dpi=500)

            #return HM

class PlotSpatialGraphmap(Algorithm):
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
    @requires("Sample", "string", ("cells",))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        samples = self.Sample[:]
        unique_samples = np.unique(self.Sample[:])
        n_cells = self.NCells[:]
        #clusters = self.Clusters[:]
        graphclusters = self.GraphCluster[:]
        n_clusters = graphclusters.max() + 1
        #cluster_ids = self.ClusterID[:]
        clusters_label_dic = {}

        labels = []
        for gi in range(n_clusters):
            if (graphclusters == gi).sum() > 50:
                #i = np.unique(clusters[graphclusters == gi])
                n = (graphclusters ==gi).sum()
                label = f" {gi:>3} ({n:,} cells) - "
                labels.append(label)
                clusters_label_dic[gi] = label
            else:
                graphclusters[graphclusters == gi] = -1
                labels.append("-1 (0 cells) ")
                clusters_label_dic[gi] = "-1 (0 cells) "
        clusters_label_dic[-1] = "-1 (0 cells) "
        labels = np.unique(labels)
        unique_colors = colorize(np.unique(graphclusters))
        unique_colors_HEX = [rgb2hex(int(color[0]*255),int(color[1]*255),int(color[2]*255)) for color in unique_colors]
        if type(self.cmap) != type(None):
            cmap = self.cmap
        else:
            cmap = {t:col for t,col in zip(labels,unique_colors_HEX)}

        for sample in unique_samples:
            filter_sample = samples == sample
            graphclusters_s = graphclusters[filter_sample]
            save_to = self.export_dir / 'Spatial_{}_{}'.format(ws._name , sample)
            if path.exists(save_to) == False:
                makedirs(save_to)


            x = self.X[:][filter_sample]
            y = self.Y[:][filter_sample]
            data = pd.DataFrame({'x': x, 'y':y, 'cluster':graphclusters_s}) #np.concatenate([xy,clusters[:,np.newaxis]],axis=1)
            #data.loc[:,'cluster'] = np.array([clusters_label_dic[c] for c in data['cluster']])
            
            xlength = x.max() - x.min()
            ylength = y.max() - y.min()
            xyratio = xlength/ylength

            if self.backend == 'holoviews':
                dicND = {}
                dicNDshortlabels = {}
                for cluster, d in data.groupby('cluster'):
                    scatter = hv.Scatter(d,kdims=['x'],vdims=['y','cluster']).opts(
                                                                            color=cmap[clusters_label_dic[cluster]], 
                                                                            height=1000,
                                                                            #width=1500,
                                                                            size=self.point_size, 
                                                                            xticks=0,
                                                                            yticks=0, 
                                                                            ylabel=None,
                                                                            xlabel=None,
                                                                            xaxis=None,
                                                                            yaxis=None, 
                                                                            bgcolor='black',
                                                                            aspect='equal',
                                                                            nonselection_fill_alpha=0,
                                                                            muted_fill_alpha=0.1,
                                                                            
                                                                            )
                    dicND[clusters_label_dic[cluster]] = scatter.opts(title=clusters_label_dic[cluster])
                    dicNDshortlabels[cluster] = scatter.opts(title=clusters_label_dic[cluster])

                ND = hv.NdOverlay(dicND).opts(
                    show_legend=True,legend_limit=100,legend_position='right',
                    fontsize={'legend':5},legend_spacing=-5,
                    )
                HM = hv.HoloMap(ND)
                
                NDshortlabels = hv.NdOverlay(dicNDshortlabels).opts(
                    show_legend=True,legend_limit=100,legend_position='right',
                    fontsize={'legend':5},legend_spacing=-5,
                    legend_muted=True,
                    )

                if save:
                    hv.save(HM, save_to / (ws._name + "_graphmap_holomap.html"))
                    hv.save(NDshortlabels, save_to / (ws._name + "_graphmap.html"))
                    NDshortlabels = NDshortlabels.opts(
                        hv.opts.Scatter(
                            size=2,
                        )
                    )
                    Layout = hv.Layout([dicNDshortlabels[x].opts(size=3, height=800, width=1600) for x in dicNDshortlabels]).cols(5)
                    hv.save(Layout, save_to / (ws._name + "_graphmap.png"),dpi=5000)

            elif self.backend == 'matplotlib':
                dic = {}
                for cluster, d in data.groupby('cluster'):
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
                    dic[clusters_label_dic[cluster]] = scatter
                ND = hv.NdOverlay(dic).opts(show_legend=True,legend_limit=100,nonselection_alpha=0)
                if save:
                    hv.save(ND, save_to / (ws._name + "_graphmap.png"),dpi=500)
                    hv.save(ND, save_to / (ws._name + "_graphmap.html"),dpi=500)

            #return ND