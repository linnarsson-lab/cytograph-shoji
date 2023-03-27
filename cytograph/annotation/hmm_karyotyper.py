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


def windowed_mean2d(x: np.ndarray, n: int):
	if x.shape[1] == 0:
		return x
	y = np.zeros_like(x)
	for center in range(x.shape[1]):
		i = max(0, center - n // 2)
		j = min(len(x), center + n // 2)
		w = x[:, i:j]
		y[:, center] = np.mean(w, axis=1)
	return y


class HmmKaryotyper(Algorithm):
	"""
	Estimate the karyotype of tumor cells using an external reference
	"""
	def __init__(
		self,
		refs: List[str],  # List of shoji workspaces
		min_umis: int = 1,
		window_size: int = 200,
		n_pca_components: int = 5,
		min_clusters: int = 10,
		n_neighbors: int = 30,
		hmm_n_states: int = 5,
		hmm_persistence: float = 0.7,
		hmm_diploid_bias: float = 0.3
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
		for ref in refs:
			logging.info(f"Loading mean expression values from '{ref}'")
			ws = db[ref]
			assert isinstance(ws, shoji.WorkspaceManager)
			y_refs = ws.MeanExpression[:]
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
		y_refs = np.concatenate(y_refs_list)

		# Select only genes from autosomes, and that are non-zero in all cell types
		self.housekeeping = ((y_refs == 0).sum(axis=0) == 0) & np.isin(self.chromosome_per_gene, list(self.chromosome_starts.keys()))
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
	@requires("Start", "int64", ("genes",))
	@requires("ClusterID", "uint32", ("clusters",))
	@creates("Aneuploid", "bool", ("clusters",))
	@creates("KaryotypePloidy", "float32", ("cells", "karyotype",))
	@creates("KaryotypePloidy", "float32", ("cells", "karyotype",))
	@creates("KaryotypePosition", "float32", ("karyotype",))
	@creates("KaryotypeChromosome", "string", ("karyotype",))
	@creates("ChromosomeStart", "uint32", ("chromosomes",))
	@creates("ChromosomeLength", "uint32", ("chromosomes",))
	@creates("ChromosomeName", "string", ("chromosomes",))
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

		if self.window_size > 1:
			logging.info("Binning along the genome")
			for ch in self.chromosome_starts.keys():
				selected = (self.chromosome_per_gene == ch)
				if selected.sum() == 0:
					continue
				self.y_refs[:, selected] = windowed_mean2d(self.y_refs[:, selected], self.window_size)
				self.y_sample[:, selected] = windowed_mean2d(self.y_sample[:, selected], self.window_size)

		logging.info("Finding best reference cell type for each cell")
		self.best_ref = np.argmax(np.corrcoef(np.log(self.y_sample + 1), np.log(self.y_refs + 1))[:n_cells, -n_refs:], axis=1)

		# Calculate ploidy of each cell along the genome
		logging.info("Computing single-cell estimated ploidy")
		self.ploidy = np.zeros((n_cells, n_windows))
		for i in range(n_cells):
			self.ploidy[i, :] = 2 * (div0(self.y_sample[i, :], self.y_refs[self.best_ref[i]]).T).T

		logging.info("Computing karyotype embedding using t-SNE")
		self.pca = PCA(n_components=self.n_pca_components).fit_transform(self.ploidy)
		self.embedding = TSNE().fit(self.pca)

		logging.info("Computing metacells by clustering the manifold using cap on cluster size")
		n_cells, n_windows = y_sample.shape
		nn = NNDescent(data=self.pca, metric="euclidean", n_neighbors=self.n_neighbors, n_jobs=-1)
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
		return self

	# def plot_metacell_embedding(self, markers: List[str] = ["PTPRC", "SOX10", "PDGFRA", "MOG", "SOX2", "WIF1", "AQP4", "HES1", "DCN", "LUM", "CDK1"]):
	# 	f = h5py.File(self.sample)
	# 	gene_names = f["shoji/Gene"][:]
	# 	plt.figure(figsize=(15, 5))
	# 	plt.subplot(2, 6, 1)
	# 	cgplot.scatterc(self.embedding, s=5, c=self.labels)
	# 	plt.title("Metacells")
	# 	plt.axis("off")
	# 	for i, gene in enumerate(markers):
	# 		plt.subplot(2, 6, i + 2)
	# 		x = f["shoji/Expression"][:, gene_names == gene].flatten()[self.cell_ordering]
	# 		cgplot.scattern(self.embedding, s=5, c=x, bgval=0, max_percentile=100)
	# 		plt.axis("off")
	# 		plt.title(gene)

	# def plot_karyotype_single_cell(self, cell: int, path: Optional[str] = None):
	# 	plt.figure(figsize=(20, 2))
	# 	plt.scatter(self.gene_positions, self.ploidy[cell, :], c="red", s=2, lw=0)
	# 	plt.vlines(3.1e9, 0, 4, linestyles="dashed", color="black", lw=0.5)
	# 	plt.hlines(1, 0, 3.1e9, linestyles="dashed", color="blue", lw=1)
	# 	plt.hlines(2, 0, 3.1e9, linestyles="dashed", color="green", lw=1)
	# 	plt.hlines(3, 0, 3.1e9, linestyles="dashed", color="blue", lw=1)

	# 	for c, j, l in zip(self.chromosome_starts.keys(), self.chromosome_starts.values(), self.chromosome_lengths.values()):
	# 		plt.text(j + l / 2, 3.5, c[3:], ha="center")
	# 	for j in self.chromosome_starts.values():
	# 		plt.vlines(j, 0, 4, linestyles="dashed", color="black", lw=0.5)

	# 	plt.ylim(0, 4)
	# 	plt.xlim(0, 3.06e9)
	# 	plt.ylabel("Ploidy")
	# 	plt.title(f"Cell #{cell} karyotype")
	# 	plt.xlabel("Genomic position")

	# 	if path is not None:
	# 		plt.savefig(path)

	# def plot_metacell_karyotype(self, metacell: int):
	# 	plt.figure(figsize=(20, 2))
	# 	for i in np.where(self.labels == metacell)[0]:
	# 		plt.scatter(self.gene_positions, self.ploidy[i, :], c="pink", s=2, lw=0)
	# 	plt.scatter(self.gene_positions, np.median(self.ploidy[self.labels == metacell, :], axis=0), c="red", s=2, lw=0)
	# 	plt.scatter(self.gene_positions, self.median_ploidy[metacell, :].flatten(), c="red", s=2, lw=0)

	# 	plt.vlines(3.1e9, 0, 4, linestyles="dashed", color="black", lw=0.5)
	# 	plt.hlines(1, 0, 3.1e9, linestyles="dashed", color="blue", lw=1)
	# 	plt.hlines(2, 0, 3.1e9, linestyles="dashed", color="green", lw=1)
	# 	plt.hlines(3, 0, 3.1e9, linestyles="dashed", color="blue", lw=1)

	# 	for c, j, l in zip(self.chromosome_starts.keys(), self.chromosome_starts.values(), self.chromosome_lengths.values()):
	# 		plt.text(j + l / 2, 3.5, c[3:], ha="center")
	# 	for j in self.chromosome_starts.values():
	# 		plt.vlines(j, 0, 4, linestyles="dashed", color="black", lw=0.5)

	# 	plt.ylim(0, 4)
	# 	plt.xlim(0, 3.06e9)
	# 	plt.ylabel("Ploidy")
	# 	plt.title(f"Metacell #{metacell} karyotype")
	# 	plt.xlabel("Genomic position")

	# def plot_predicted_metacell_karyotypes(self):
	# 	n_metacells = self.labels.max() + 1
	# 	plt.figure(figsize=(20, n_metacells * 1.5))

	# 	for ix in range(n_metacells):
	# 		plt.subplot(n_metacells, 1, ix + 1)
	# 		plt.vlines(3.1e9, 0, 4, linestyles="dashed", color="black", lw=0.5, zorder=-1)
	# 		plt.hlines(1, 0, 3.1e9, linestyles="dashed", color="blue", lw=0.5, zorder=-1)
	# 		plt.hlines(2, 0, 3.1e9, linestyles="dashed", color="green", lw=0.5, zorder=-1)
	# 		plt.hlines(3, 0, 3.1e9, linestyles="dashed", color="blue", lw=0.5, zorder=-1)

	# 		plt.scatter(self.gene_positions, self.predicted_ploidy[ix], c="lightgreen", s=10, lw=0)
	# 		plt.scatter(self.gene_positions, self.median_ploidy[ix], c="red", s=1, lw=0)

	# 		for c, j, l in zip(self.chromosome_starts.keys(), self.chromosome_starts.values(), self.chromosome_lengths.values()):
	# 			plt.text(j + l / 2, 3.5, c[3:], ha="center")
	# 		for j in self.chromosome_starts.values():
	# 			plt.vlines(j, 0, 4, linestyles="dashed", color="black", lw=0.5)

	# 		plt.ylim(0, 4)
	# 		plt.xlim(0, 3.06e9)
	# 		plt.ylabel("Ploidy")
	# 		plt.text(0, 0.5, f"Metacell #{ix}")
	# 		plt.xticks([])

