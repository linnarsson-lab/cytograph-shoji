import logging
from typing import List, Optional

import fastcluster
import igraph
import leidenalg as la
#import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hc
from hmmlearn import hmm
from openTSNE import TSNE
from pynndescent import NNDescent
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

import shoji
#import cytograph.visualization as cgplot
from ..utils import div0
from ..algorithm import Algorithm, creates, requires

import logging
from typing import List, Optional

import fastcluster
import igraph
import leidenalg as la
import numpy as np
import scipy.cluster.hierarchy as hc
from hmmlearn import hmm
from openTSNE import TSNE
from pynndescent import NNDescent
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

import shoji
from cytograph.utils import div0
from cytograph.algorithm import Algorithm, creates, requires



def windowed_mean2d(x: np.ndarray, n: int):
    if x.shape[1] == 0:
        return x
    length = x.shape[1]
    y = np.zeros_like(x)
    for center in range(length):
        i = max(0, center - n // 2)
        j = min(length, center + n // 2)
        
        w = x[:, i:j]
        y[:, center] = np.mean(w, axis=1)
    return y

def windowed_median2d(x: np.ndarray, n: int):
    if x.shape[1] == 0:
        return x
    length = x.shape[1]
    y = np.zeros_like(x)
    for center in range(length):
        i = max(0, center - n // 2)
        j = min(length, center + n // 2)
        
        w = x[:, i:j]
        y[:, center] = np.median(w, axis=1)
    return y


class HmmKaryotyper(Algorithm):
    """
    Estimate the karyotype of tumor cells using an external reference
    """
    def __init__(
        self,
        refs: List[str],  # List of shoji workspaces
        min_umis: int = 1,
        window_size: int = 300,
        n_pca_components: int = 5,
        min_clusters: int = 10,
        n_neighbors: int = 30,
        hmm_n_states: int = 5,
        hmm_persistence: float = 0.5,
        hmm_diploid_bias: float = 0.35
    ):
        self.refs = refs
        self.min_umis = min_umis
        self.window_size = window_size
        self.n_pca_components = n_pca_components
        self.min_clusters = min_clusters
        self.n_neighbors = n_neighbors

        self.hmm_n_states = hmm_n_states
        self.hmm_persistence = hmm_persistence
        self.hmm_diploid_bias = hmm_diploid_bias

        self.chromosome_starts = {
            'chr1': 0,
            'chr2': 248956422,
            'chr3': 491149951,
            'chr4': 689445510,
            'chr5': 879660065,
            'chr6': 1061198324,
            'chr7': 1232004303,
            'chr8': 1391350276,
            'chr9': 1536488912,
            'chr10': 1674883629,
            'chr11': 1808681051,
            'chr12': 1943767673,
            'chr13': 2077042982,
            'chr14': 2191407310,
            'chr15': 2298451028,
            'chr16': 2400442217,
            'chr17': 2490780562,
            'chr18': 2574038003,
            'chr19': 2654411288,
            'chr20': 2713028904,
            'chr21': 2777473071,
            'chr22': 2824183054,
            'chrX': 2875001522,
            'chrY': 3031042417
        }
        self.chromosome_lengths = {
            'chr1': 248956422,
            'chr2': 242193529,
            'chr3': 198295559,
            'chr4': 190214555,
            'chr5': 181538259,
            'chr6': 170805979,
            'chr7': 159345973,
            'chr8': 145138636,
            'chr9': 138394717,
            'chr10': 133797422,
            'chr11': 135086622,
            'chr12': 133275309,
            'chr13': 114364328,
            'chr14': 107043718,
            'chr15': 101991189,
            'chr16': 90338345,
            'chr17': 83257441,
            'chr18': 80373285,
            'chr19': 58617616,
            'chr20': 64444167,
            'chr21': 46709983,
            'chr22': 50818468,
            'chrX': 156040895,
            'chrY': 8957583
        }

        logging.info("Preparing the references")
        # Load the references
        db = shoji.connect()
        y_refs_list = []
        shared_genes = np.ones(db[refs[0]].genes.length, dtype=bool)
        for ref in refs:
            logging.info(f"Loading mean expression values from '{ref}'")
            ws = db[ref]
            assert isinstance(ws, shoji.WorkspaceManager)
            y_refs = np.nan_to_num(ws.MeanExpression[:])
            shared_genes = shared_genes & (np.isnan(ws.MeanExpression[:]).sum(axis=0) == 0)
            logging.info(f"{shared_genes.sum()} shared genes")
            assert isinstance(y_refs, np.ndarray)
            totals = y_refs.sum(axis=1)
            self.std_size = np.median(totals)
            y_refs = (y_refs.T / totals * self.std_size).T
            y_refs_list.append(y_refs)

            if len(y_refs_list) == 1:
                logging.info(f"Loading genes from '{ref}'")
                self.accessions = ws.Accession[:]
                self.gene_positions = np.array([int(s) for s in ws.Start[:]])  # type: ignore
                self.chromosome_per_gene = ws.Chromosome[:]
            else:
                assert np.all(self.accessions == ws.Accession[:]), f"Genes in {ref} do not match (by accessions or ordering) those of {refs[0]}"  # type: ignore
        self.y_refs = np.concatenate(y_refs_list)

        # Select only genes from autosomes, and that are >10% non-zero and not NaN in all cell types
        self.housekeeping = shared_genes & (np.count_nonzero(self.y_refs, axis=0) > self.y_refs.shape[0] / 10) & np.isin(self.chromosome_per_gene, list(self.chromosome_starts.keys()))
        self.y_refs = self.y_refs[:, self.housekeeping]
        self.chromosome_per_gene = self.chromosome_per_gene[self.housekeeping]
        self.gene_positions = self.gene_positions[self.housekeeping]
        logging.info(f"Selected {self.housekeeping.sum()} housekeeping genes")

        # Convert gene positions to genome-wide coordinates
        for ch, offset in self.chromosome_starts.items():
            self.gene_positions[self.chromosome_per_gene == ch] += offset
        # Order genes by genomic position
        self.gene_ordering = np.argsort(self.gene_positions)
        self.chromosome_per_gene = self.chromosome_per_gene[self.gene_ordering]
        self.gene_positions = self.gene_positions[self.gene_ordering]
        self.y_refs = self.y_refs[:, self.gene_ordering]

        self.chromosome_borders = []
        ix = 0
        for ch in self.chromosome_starts.keys():
            n = (self.chromosome_per_gene == ch).sum()
            self.chromosome_borders.append(n + ix)
            ix += n
        
    def _compute_metacell_dendrogram(self):
        logging.info("Computing Ward's linkage dendrogram of metacells")
        n_cells = self.ploidy.shape[0]
        n_metacells = self.median_ploidy.shape[0]
        D = pdist(self.median_ploidy, 'correlation')
        Z = fastcluster.linkage(D, 'ward', preserve_input=True)
        Z = hc.optimal_leaf_ordering(Z, D)
        metacell_ordering = hc.leaves_list(Z)

        # Renumber the linkage matrix so it corresponds to the reordered clusters
        for i in range(len(Z)):
            if Z[i, 0] < n_metacells:
                Z[i, 0] = np.where(Z[i, 0] == metacell_ordering)[0]
            if Z[i, 1] < n_metacells:
                Z[i, 1] = np.where(Z[i, 1] == metacell_ordering)[0]

        # Renumber the labels and sort the cells according to the new ordering
        new_labels = np.zeros_like(self.labels)
        self.cell_ordering = np.zeros(n_cells, dtype="int")
        offset = 0
        for ix, lbl in enumerate(metacell_ordering):
            selected = (self.labels == lbl)
            n_selected = selected.sum()
            self.cell_ordering[offset:offset + n_selected] = np.where(selected)[0]
            new_labels[offset:offset + n_selected] = ix
            offset += n_selected
        self.labels = new_labels

        self.y_sample = self.y_sample[self.cell_ordering]
        self.ploidy = self.ploidy[self.cell_ordering]
        self.median_ploidy = self.median_ploidy[metacell_ordering]
        self.embedding = self.embedding[self.cell_ordering, :]
        self.pca = self.pca[self.cell_ordering, :]
        self.best_ref = self.best_ref[self.cell_ordering]
        self.dendrogram = Z

    @requires("GeneTotalUMIs", "uint32", ("genes",))
    @requires("Chromosome", "string", ("genes",))
    @requires("Start", "string", ("genes",))
    @requires("ClusterID", "uint32", ("clusters",))
    @requires("Expression", "uint16", ("cells", "genes"))
    @creates("KaryotypeBestReference", "uint32", ("cells",))
    @creates("KaryotypeCellOrdering", "uint32", ("cells",))
    @creates("KaryotypeDendrogram", "float64", (None, 4))
    @creates("KaryotypeEmbedding", "float32", ("cells", 2))
    @creates("KaryotypeHousekeepingGenes", "bool", ("genes",)) 
    @creates("KaryotypeGeneOrdering", "uint32", ("karyotype_genes",)) 
    @creates("KaryotypeGenePositions", "uint64", ("karyotype_genes",))
    @creates("KaryotypeMetacells", "uint32", ("cells",)) 
    @creates("KaryotypeMedianPloidy", "float32", ("karyotype_metacells", "karyotype_windows"))
    @creates("KaryotypePloidy", "float32", ("cells", "karyotype_windows")) 
    @creates("KaryotypePredictedPloidy", "uint16", ("karyotype_metacells", "karyotype_windows"))
    @creates("KaryotypeChromosomeBorders", "uint32", (None,))
    def fit(self, ws, save: bool = False):
        n_refs, n_windows = self.y_refs.shape

        logging.info("Loading the sample")
        y_sample = self.Expression[:]
        assert isinstance(y_sample, np.ndarray)
        assert np.all(ws.Accession[:] == self.accessions), "Genes in sample do not match (by accessions or ordering) those of the reference"
        totals = y_sample.sum(axis=1)
        y_sample = (y_sample.T / totals * self.std_size).T
        y_sample = y_sample[:, self.housekeeping]
        self.y_sample = y_sample[:, self.gene_ordering]
        n_cells, _ = self.y_sample.shape
        logging.info(f"Loaded {n_cells} cells")

        logging.info("Finding best reference cell type for each cell")
        # self.best_ref = np.argmax(np.corrcoef(np.log(self.y_sample + 1), np.log(self.y_refs + 1))[:n_cells, -n_refs:], axis=1)
        temp = []
        logged_refs = np.log(self.y_refs + 1)
        for ix in range(n_cells):
            temp.append(np.argmax(np.corrcoef(np.log(self.y_sample[ix, :] + 1), logged_refs)[0, 1:]))
        self.best_ref = np.array(temp)

        y_refs = self.y_refs.copy()
        if self.window_size > 1:
            logging.info("Binning along the genome")
            for ch in self.chromosome_starts.keys():
                selected = (self.chromosome_per_gene == ch)
                if selected.sum() == 0:
                    continue
                y_refs[:, selected] = windowed_mean2d(self.y_refs[:, selected], self.window_size)
                self.y_sample[:, selected] = windowed_mean2d(self.y_sample[:, selected], self.window_size)
        
        # Calculate ploidy of each cell along the genome
        logging.info("Computing single-cell estimated ploidy")
        self.ploidy = np.zeros((n_cells, n_windows))
        for i in range(n_cells):
            self.ploidy[i, :] = 2 * (div0(self.y_sample[i, :], y_refs[self.best_ref[i]]).T).T
        self.ploidy = windowed_median2d(self.ploidy, int(self.window_size * 3))
        
        logging.info("Computing karyotype embedding using t-SNE")
        self.pca = PCA(n_components=self.n_pca_components).fit_transform(self.ploidy)
        self.embedding = TSNE().fit(self.pca)

        logging.info("Computing approximate nearest neighbours")
        n_cells, n_windows = y_sample.shape
        nn = NNDescent(data=self.pca, metric="euclidean", n_neighbors=self.n_neighbors, n_jobs=-1)

        logging.info("Computing metacells by clustering the manifold using cap on cluster size")
        edges = []
        assert nn.neighbor_graph is not None
        for i in range(n_cells):
            for j in range(nn.neighbor_graph[0].shape[1]):
                edges.append((i, nn.neighbor_graph[0][i, j]))
        manifold = igraph.Graph(n_cells, edges, directed=False)
        self.labels = np.array(la.find_partition(manifold, la.ModularityVertexPartition, max_comm_size=n_cells // self.min_clusters, n_iterations=-1).membership)
        n_metacells = self.labels.max() + 1

        # Calculate median ploidy of each metacell along the genome
        logging.info("Computing metacell median ploidy")
        self.median_ploidy = np.zeros((n_metacells, n_windows))
        for lbl in np.unique(self.labels):
            self.median_ploidy[lbl, :] = np.median(self.ploidy[self.labels == lbl, :], axis=0)

        # Calculate dendrogram and reorder all the tensors
        self._compute_metacell_dendrogram()

        logging.info("Inferring karyotype using gaussian HMM")
        # All parameters are fixed, not learned from the data
        m = hmm.GaussianHMM(n_components=self.hmm_n_states, init_params="", params="")
        # n_components is the number of HMM states
        # n_features is the number of dimensions of the multivariate gaussian emmissions

        # Start in the diploid state
        m.startprob_ = np.full(self.hmm_n_states, (0.5 / (self.hmm_n_states - 1)))  # (n_components, )
        m.startprob_[2] = 0.5

        # Transitions are biased towards staying in the current state with probability equal to persistence
        # and is more likely to return to diploid than any of the other states (given by diploid_bias)
        m.transmat_ = np.full((5, 5), (1 - self.hmm_persistence - self.hmm_diploid_bias) / (self.hmm_n_states - 1))  # (n_components, n_components)
        np.fill_diagonal(m.transmat_, self.hmm_persistence)
        m.transmat_[:, 2] += self.hmm_diploid_bias

        # Used fixed means (set to integer levels of ploidy (0 to n_states) and covariances
        m.means_ = np.arange(self.hmm_n_states)[:, None]  # (n_components, n_features)
        covars = np.full((1, self.hmm_n_states), 0.05)
        covars[0, 2] = 0.25
        covars[0, -1] = 1.0
        m.covars_ = covars

        # Predict ploidy along the chromosomes
        # Need to flatten to make hmmlearn happy
        median_ploidy = self.median_ploidy.flatten()[:, None]  # (n_samples, n_features)
        n_samples = [n_windows] * n_metacells  # (n_sequences,); sum should be n_samples
        m.fit(median_ploidy, lengths=[n_samples])
        predicted_ploidy = m.predict(median_ploidy)
        # Reshape back to original shape
        self.predicted_ploidy = predicted_ploidy.reshape((n_metacells, n_windows))
        if save:
            del ws.karyotype_genes
            del ws.karyotype_metacells
            del ws.karyotype_windows
            del ws.KaryotypeDendrogram
            del ws.KaryotypeMedianPloidy
            del ws.KaryotypePloidy
            del ws.KaryotypePredictedPloidy
            ws.karyotype_metacells = shoji.Dimension(n_metacells)
            ws.karyotype_windows = shoji.Dimension(n_windows)
            ws.karyotype_genes = shoji.Dimension(self.housekeeping.sum())
        return (
            self.best_ref.astype("uint32"),
            self.cell_ordering.astype("uint32"),
            self.dendrogram,
            self.embedding,
            self.housekeeping, 
            self.gene_ordering.astype("uint32"), 
            self.gene_positions.astype("uint64"), 
            self.labels.astype("uint32"), 
            self.median_ploidy.astype("float32"), 
            self.ploidy.astype("float32"), 
            self.predicted_ploidy.astype("float32"),
            np.array(self.chromosome_borders, dtype="uint32")
        )
