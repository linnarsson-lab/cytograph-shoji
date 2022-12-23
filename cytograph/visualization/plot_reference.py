import scanpy as sc
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scvi

import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram
import fastcluster
import scipy
import shoji
from cytograph import Algorithm, Species, requires
import logging
from matplotlib import rcParams

from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
rcParams['pdf.fonttype'] = 42 # enables correct plotting of text for PDFs

cluster_colors_GBM = {
    'AC-like':'#c2f970',#inchworm 
    'MES-like':'#cf2f74',# Deep cerise
    'NPC-like':'#e6ff6e',#laser lemon
    'RG':'#8c14fc', #electric indigo
    'OPC-like':'#29f1c3', #bright turquoise
    
    'Astrocyte':'#26c281', #jungles greeen
    'OPC':'#038aff',#azure radiance
    'Neuron':'#f3e16b',#golden sand
    'Oligodendrocyte':'#cdd1e4', #lavander grey
    
    'B cell':'#f9690e',#ecstasy 
    'Plasma B':'#f27935', #jaffa
    'CD4/CD8':'#fabe58', #saffron mango
    'DC':'#ff9470', #atomic tangerine 
    'Mast':'#af4154', #hippie pink
    'Mono':'#f22613', #scarlet
    'TAM-BDM':'#f64747', #sunset orange
    'TAM-MG':'#ff4c30',#red orange
    'NK':'#a37c82', #pharlap
    
    'Endothelial':'#d5b8ff',  #mauve
    'Mural cell': '#f1e7fe',  #magnolia 
}

class PlotReference(Algorithm):
    def __init__(
        self, 
        reference_csv:str,
        selected_genes:list,
        cluster_colors:dict=None,
        filename: str = "reference.png",

        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.filename = filename
        self.reference_csv = reference_csv
        self.selected_genes = selected_genes
        self.cluster_colors = cluster_colors

    @requires("MeanExpression", "float64", ("clusters", "genes"))
    @requires("Gene", "string", ("genes",))
    @requires("Enrichment", "float32", ("clusters", "genes"))
    @requires("Species", "string", ())
    @requires("NCells", "uint64", ("clusters",))
    @requires("Clusters", "uint32", ("cells",))
    @requires("ClusterID", "uint32", ("clusters",))
    @requires("GraphCluster", "int8", ("cells",))
    @requires("Embedding", "float32", ("cells", 2))
    @requires("Sample", "string", ("cells",))
    @requires("Expression", "uint16", ("cells","genes"))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        sample = ws.Sample[:]
        gene = self.Gene[:]
        unique_samples = np.unique(sample)
        graphcluster = self.GraphCluster[:]
        unique_graphclusters = np.unique(graphcluster)
        expression = self.Expression[:]

        sample_graphclusters = np.array([np.zeros(80) for x in range(unique_samples.shape[0])])

        mean_expression_gc = []
        for gc in unique_graphclusters:
            expression_gc = expression[graphcluster==gc]
            expression_gc = expression_gc.sum(axis=0)
            mean_expression_gc.append(expression_gc)
        mean_expression_gc = np.stack(mean_expression_gc)
        logging.info('GraphClusters shape {}'.format(mean_expression_gc.shape))
        
        eel_means = sc.AnnData(X=mean_expression_gc, obs=pd.DataFrame({'cluster':unique_graphclusters}), var=pd.DataFrame(index=gene))
        eel_means.var_names_make_unique()
        df_ref = pd.read_csv(self.reference_csv, index_col=0)
        ref_means = sc.AnnData(X=df_ref.values.T,obs=pd.DataFrame({'cell_types':df_ref.columns}),var=pd.DataFrame(index=df_ref.index))
        ref_means.var_names_make_unique()

        adata_sc_means = ref_means[:,self.selected_genes]
        adata_spatial_means = eel_means[:,self.selected_genes]
        sc.pp.log1p(adata_sc_means)
        sc.pp.log1p(adata_spatial_means)
        sc.pp.scale(adata_sc_means)
        sc.pp.scale(adata_spatial_means)

        correlation_matrix = np.zeros([adata_sc_means.shape[0], adata_spatial_means.shape[0]])
        for seq_c in range(adata_sc_means.shape[0]):
            for sp_c in range(adata_spatial_means.shape[0]):
                corr = scipy.stats.pearsonr(adata_sc_means.X[seq_c,:], adata_spatial_means.X[sp_c,:])
                #scipy.spatial.distance.cdist()
                correlation_matrix[seq_c,sp_c] = corr.statistic
                
        df = pd.DataFrame(data=correlation_matrix.T,index=adata_spatial_means.obs.GraphCluster,columns=adata_sc_means.obs.annotation_level_3)

        # Ordering scRNAseq
        D = pdist(df.values.T, 'euclidean')
        Z = fastcluster.linkage(D, 'complete', preserve_input=True)
        Z = hc.optimal_leaf_ordering(Z, D, metric='euclidean')
        ordering_sc = hc.leaves_list(Z)
        ordering_sc_str = df.columns[ordering_sc]

        # Ordering EEL
        D = pdist(df.values, 'euclidean')
        Z = fastcluster.linkage(D, 'complete', preserve_input=True)
        Z = hc.optimal_leaf_ordering(Z, D, metric='euclidean')
        ordering_sp = hc.leaves_list(Z)
        ordering_sp_str = df.index.values[ordering_sp]

        df = df.loc[ordering_sp_str,ordering_sc_str]

        heat_map(df, 
         df.columns,sort=ordering_sc_str.values,
         cluster_colors=cluster_colors_GBM
        )
        plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=1000, bbox_inches='tight')
        plt.close()

def heat_map(df, labels, sort=None, cluster_colors=None, 
             cluster_number = False, save=False, name='', fontsz=8):
    """
    Plot heat_map of a dataframe.
    Input:
    `df`(pandas dataframe): Dataframe to plot. Cells as columns, genes as rows.
    `labels`(list/array): list of labels of the cells in the same order as the 
        df columns.
    `sort`(list): List of sorted cluster numbers. If None it will plot them in
        assending order
    `cluster_colors`(dict): Dictionary matching cluster numbers with colors
        in the hex fomat. Like '#ff0000' for red
    `cluster_number`(bool): Show cluster numbers in the top color bar. Usefull
        when manually sorting the clusters.
    `save`(bool): option to save the figure as png with 300dpi.
    `name`(str): Name to use when saving
    
    """
    #Find the name of the input df, for logging
    df_input_name =[x for x in globals() if globals()[x] is df][0]
    #print('df used for plot: {}'.format(df_input_name))
    
    if type(sort) == type(None):
        optimal_order = np.unique(labels)    
    else:
        optimal_order = sort
    #print('Order of clusters: {}'.format(optimal_order))
    
    cl, lc = gen_labels(df, np.array(labels))[:2]

    
    #Sort the cells according to the optimal cluster order
    optimal_sort_cells = sum([lc[i] for i in optimal_order], [])
    
    #Create a list of optimal sorted cell labels
    optimal_sort_labels = [cl[i] for i in optimal_sort_cells]
    
    fig, axHM = plt.subplots(figsize=(20,10))
    
    df_full = df.copy()
    z = df_full.values
    z = pd.DataFrame(z, index=df_full.index, columns=df_full.columns)
    z = z.loc[:,optimal_sort_cells]
    
    
    im = axHM.pcolormesh(z.values, cmap='magma')
    
    plt.yticks(np.arange(0.5, len(df.index), 1), z.index, fontsize=fontsz)
    plt.gca().invert_yaxis()
    plt.xlim(xmax=len(labels))
    plt.xticks(np.arange(0.5,df.shape[1],step=1), labels=z.columns, rotation='vertical', fontsize=fontsz)

    divider = make_axes_locatable(axHM)
    axLabel = divider.append_axes("top", .3, pad=0, sharex=axHM)
    
    counter = Counter(labels)
    pos=0
    if cluster_colors == None:
        optimal_sort_labels = np.arange(len(optimal_sort_labels))# np.array(
        print(optimal_sort_labels)
        axLabel.pcolor(optimal_sort_labels[None,:]/max(optimal_sort_labels), cmap='nipy_spectral')
        if cluster_number==True:
            for l in optimal_order:
                axLabel.text(pos + (counter[l]/2), 1.2, l, fontsize=fontsz,
                         horizontalalignment='center', verticalalignment='center')
                pos += Counter(labels)[l]
        
    else:
        for l in optimal_order:
            #Use Bottom instead of y for older versions of matplotlib
            axLabel.barh(y = 0, left = pos, width = counter[l], color=cluster_colors[l],
                        linewidth=0.5, edgecolor=cluster_colors[l])
            if cluster_number==True:
                axLabel.text(pos + (counter[l]/2), 0, l, fontsize=fontsz,
                         horizontalalignment='center', verticalalignment='center')
            pos += Counter(labels)[l]
    
    axLabel.set_xlim(xmax=len(labels))
    axLabel.axis('off')
    
    cax = fig.add_axes([.91, 0.13, 0.01, 0.22])
    colorbar = fig.colorbar(im, cax=cax, ticks=[0,1])
    colorbar.set_ticklabels(['0', 'max'])
    
def gen_labels(df, model):
    """
    Generate cell labels from model.
    Input:
    `df`: Panda's dataframe that has been used for the clustering. (used to 
    get the names of colums and rows)
    `model`(obj OR array): Clustering object. OR numpy array with cell labels.
    Returns (in this order):
    `cell_labels` = Dictionary coupling cellID with cluster label
    `label_cells` = Dictionary coupling cluster labels with cellID
    `cellID` = List of cellID in same order as labels
    `labels` = List of cluster labels in same order as cells
    `labels_a` = Same as "labels" but in numpy array
    
    """
    
    if str(type(model))[0] == "":
        cell_labels = dict(zip(df.columns, model.labels_))
        label_cells = {}
        for l in np.unique(model.labels_):
            label_cells[l] = []
        for i, label in enumerate(model.labels_):
            label_cells[label].append(df.columns[i])
        cellID = list(df.columns)
        labels = list(model.labels_)
        labels_a = model.labels_
    elif type(model) == np.ndarray:
        cell_labels = dict(zip(df.columns, model))
        label_cells = {}
        for l in np.unique(model):
            label_cells[l] = []
        for i, label in enumerate(model):
            label_cells[label].append(df.columns[i])
        cellID = list(df.columns)
        labels = list(model)
        labels_a = model
    else:
        print('Error wrong input type')
    
    return cell_labels, label_cells, cellID, labels, labels_a
