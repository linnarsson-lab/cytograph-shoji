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
import holoviews as hv

def indices_to_order_a_like_b(a, b):
    return a.argsort()[b.argsort().argsort()]

def plot_clusters(ws, save=False, size=1, cmap=None,filename='clusters',backend='matplotlib', format='png'):
    hv.extension(backend)
    xy = ws.Embedding[:]
    clusters = ws.Clusters[:]

    clusters = ws.Clusters[:]#[subsample]
    n_clusters = clusters.max() + 1

    labels = []
    ordering = indices_to_order_a_like_b(ws.ClusterID[:], np.arange(n_clusters))
    ann_names = ws.AnnotationName[:]
    ann_post = ws.AnnotationPosterior[:].T[:, ordering]
    clusters_label_dic = {}
    for i in range(ws.clusters.length):
        order = ann_post[:,i].argsort()[::-1]
        label = ann_names[order[0]]
        #label = str(i)+ ' - ' + ' | '.join(label)
        clusters_label_dic[i] = label
        labels.append(label)
    labels = np.unique(labels)

    unique_colors = colorize(np.unique(clusters))
    unique_colors_HEX = [rgb2hex(int(color[0]*255),int(color[1]*255),int(color[2]*255)) for color in unique_colors]
    if type(cmap) != type(None):
        cmap=cmap
    else:
        cmap = {t:col for t,col in zip(labels,unique_colors_HEX)}
    data = pd.DataFrame({'x':xy[:,0], 'y':xy[:,1], 'cluster':clusters}) #np.concatenate([xy,clusters[:,np.newaxis]],axis=1)

    subsample = np.random.choice(np.arange(len(data)), size=100000, replace=False)
    #data = data[subsample,:]
    data.loc[:,'cluster'] = np.array([clusters_label_dic[c] for c in data['cluster']])
    if backend == 'matplotlib':
        dic = {}
        for cluster, d in data.groupby('cluster'):
            print(cluster)
            
            scatter = hv.Scatter(d,kdims=['x'],vdims=['y','cluster']).opts(
                                                                    color=cmap[cluster], 
                                                                    fig_size=800,
                                                                    s=size, 
                                                                    xticks=0,
                                                                    yticks=0, 
                                                                    ylabel=None,
                                                                    xlabel=None,
                                                                    xaxis=None,
                                                                    yaxis=None, 
                                                                    )
            dic[cluster] = scatter
        scatter = hv.NdOverlay(dic).opts(show_legend=True)
    else:
        dic = {}
        for cluster, d in data.groupby('cluster'):
            print(cluster)

            scatter = hv.Scatter(d,kdims=['x'],vdims=['y','cluster']).opts(
                                                                    color=cmap[cluster], 
                                                                    width=800, 
                                                                    height=800,
                                                                    size=size, 
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
        hv.save(scatter, filename+'.{}'.format(format),dpi=500)

    return scatter

def plot_gene(ws, gene, save=False):
    hv.extension('matplotlib')
    xy = ws.Embedding[:]
    exp = ws.Expression[:,ws.Gene[:] == gene]
    d = np.concatenate([xy,exp],axis=1)
    zeros = exp <= 2
    s0 = d[zeros[:,0],:]
    s0 = hv.Scatter(s0,kdims=['x'],vdims=['y','z']).opts(color='grey',s=1, alpha=0.01)

    s1 = d[~zeros[:,0],:]
    s1[:,2] = np.log(s1[:,2]+1)
    s1 = hv.Scatter(s1,kdims=['x'],vdims=['y','z']).opts(color=hv.dim('z'),cmap='fire',s=2,alpha=1)

    s= s0*s1
    s = s.opts(xticks=0,yticks=0, ylabel=None,xlabel=None,xaxis=None,yaxis=None, fig_size=200)
    if save:
        hv.save(s, '{}.png'.format(gene))
    return s