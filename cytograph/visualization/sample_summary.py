import logging
import matplotlib.pyplot as plt
import numpy as np
import shoji
from cytograph import Algorithm, Species, requires
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
import fastcluster
from scipy.cluster.hierarchy import dendrogram
from .colors import colorize


class SampleBarplot(Algorithm):
    def __init__(
        self, 
        filename: str = "sample_barplot.png", 
        **kwargs
        ) -> None:
        super().__init__(**kwargs)
        self.filename = filename


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
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        sample = ws.Sample[:]
        unique_samples = np.unique(sample)
        graphcluster = ws.GraphCluster[:]

        sample_graphclusters = np.array([np.zeros(80) for x in range(unique_samples.shape[0])])

        for s, r in zip( unique_samples,range( unique_samples.shape[0])):
            graphcluster_s = graphcluster[sample == s]
            i, counts = np.unique(graphcluster_s, return_counts=True)
            sample_graphclusters[r,i]= counts

        fig = plt.figure(figsize=(10,10), dpi=500)

        fig_spec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1,2])
        fig_spec.hspace = 0.1
        fig_spec.wspace = 0
        subplot = 0
        ax = fig.add_subplot(fig_spec[subplot])

        x = (sample_graphclusters.T/sample_graphclusters.sum(axis=1)).T
        D = pdist(x, 'euclidean')
        Z = fastcluster.linkage(D, 'ward', preserve_input=True)
        Z = hc.optimal_leaf_ordering(Z, D, metric='euclidean')
        ordering = hc.leaves_list(Z)
        ordering_str = unique_samples[ordering]
        lines = dendrogram(
                    Z, 
                    #labels=ordering,
                    color_threshold=0.1,
                    above_threshold_color='black',
                    #link_color_func=lambda k: colors[k]
                    ax=ax
            )

        ax.set_ylim(0, Z[:, 2].max() * 2)
        ax.axis('off')

        width = 0.8
        subplot = 1
        ax = fig.add_subplot(fig_spec[subplot])
        indexes = np.arange(sample_graphclusters.shape[0])
        unique_colors = colorize(np.arange(sample_graphclusters.shape[1]))

        sample_graphclusters_norm = (sample_graphclusters.T/sample_graphclusters.sum(axis=1)).T * 100
        organize_clusters = [np.where(unique_samples==x)[0][0] for x in ordering_str]
        sample_graphclusters_norm = sample_graphclusters_norm[organize_clusters,:]

        loc = np.zeros(sample_graphclusters.shape[0])
        for cluster in range(sample_graphclusters.shape[1]):

            i_ = sample_graphclusters_norm[:,cluster]
            #print(i_.shape)

            ax.bar(indexes, i_[ordering], width, bottom=loc,color=unique_colors[cluster])
            loc += i_[ordering]

        ax.set_xticks(np.arange(ordering_str.shape[0]), ordering_str,)
        ax.tick_params(axis='both', which='major', labelsize=5)
        #ax.get_yaxis().set_ticks([])
        for pos in ['right', 'top', 'bottom', 'left']:
            ax.spines[pos].set_visible(False)

        ax.set_xlim(-0.5, ordering.shape[0] - 0.5)
        fig.legend(labels=np.arange(-1, sample_graphclusters.shape[1]), 
                loc='lower center', 
                fontsize=5,
                #bbox_to_anchor=(0, 0),
                ncol=20,
                frameon=False,
                )

        plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=1000, bbox_inches='tight')
        plt.close()

###

