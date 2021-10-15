# This function written by Kimberly Siletti and is based on doubletFinder.R as forwarded by the Allen Institute:
#
# "Doublet detection in single-cell RNA sequencing data
#
# This function generates artificial nearest neighbors from existing single-cell RNA
# sequencing data. First, real and artificial data are merged. Second, dimension reduction
# is performed on the merged real-artificial dataset using PCA. Third, the proportion of
# artificial nearest neighbors is defined for each real cell. Finally, real cells are rank-
# ordered and predicted doublets are defined via thresholding based on the expected number
# of doublets.
#
# @param seu A fully-processed Seurat object (i.e. after normalization, variable gene definition,
# scaling, PCA, and tSNE).
# @param expected.doublets The number of doublets expected to be present in the original data.
# This value can best be estimated from cell loading densities into the 10X/Drop-Seq device.
# @param porportion.artificial The proportion (from 0-1) of the merged real-artificial dataset
# that is artificial. In other words, this argument defines the total number of artificial doublets.
# Default is set to 25%, based on optimization on PBMCs (see McGinnis, Murrow and Gartner 2018, BioRxiv).
# @param proportion.NN The proportion (from 0-1) of the merged real-artificial dataset used to define
# each cell's neighborhood in PC space. Default set to 1%, based on optimization on PBMCs (see McGinnis,
# Murrow and Gartner 2018, BioRxiv).
# @return An updated Seurat object with metadata for pANN values and doublet predictions.
# @export
# @examples
# seu <- doubletFinder(seu, expected.doublets = 1000, proportion.artificial = 0.25, proportion.NN = 0.01)"


import logging
from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import shoji
from cytograph.enrichment import FeatureSelectionByVariance
from cytograph import requires, creates, Module
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from unidip import UniDip


class DoubletFinder(Module):
	def __init__(self, proportion_artificial: float = 0.2, fixed_threshold: float = None, max_threshold: float = 1, k: int = None, **kwargs) -> None:
		super().__init__(**kwargs)
		self.proportion_artificial = proportion_artificial
		self.fixed_threshold = fixed_threshold
		self.max_threshold = max_threshold
		self.k = k

	@requires("Expression", None, ("cells", "genes"))
	@creates("DoubletScore", "float32", ("cells",))
	@creates("DoubletFlag", "bool", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Find doublets using the doublet-finder algorithm.

		Args:
			ws		Workspace
			save	If true, save the result to the workspace

		Returns:
			DoubletScore		A doublet score in the interval [0, 1]
			DoubletFlag			0: singlet, 1: doublet, 2: neighbor of a doublet
		"""
		# WARNING: for historical reasons, all the processing here is done with matrices oriented (genes, cells)
		# Step 1: Generate artificial doublets from input
		logging.info(" DoubletFinder: Creating artificial doublets")
		n_real_cells = ws.cells.length
		n_genes = ws.genes.length
		n_doublets = int(n_real_cells / (1 - self.proportion_artificial) - n_real_cells)
		doublets = np.zeros((n_genes, n_doublets))
		expression = self.Expression[:]
		for i in range(n_doublets):
			(a, b) = np.random.choice(n_real_cells, size=2, replace=False)
			doublets[:, i] = expression[a] + expression[b]

		data_wdoublets = np.concatenate((expression.T, doublets), axis=1)  # Transpose the expression matrix to be (genes, cells)
		logging.info(" DoubletFinder: Feature selection and dimensionality reduction")
		genes = FeatureSelectionByVariance(2000).fit(ws)
		logging.info(" DoubletFinder: Computing size factors")
		f = np.divide(data_wdoublets.sum(axis=0), 10e4)
		logging.info(" DoubletFinder: Normalizing by size factors")
		norm_data = np.divide(data_wdoublets, f)
		logging.info(" DoubletFinder: Log transforming")
		norm_data = np.log(norm_data + 1)
		logging.info(" DoubletFinder: PCA to 50 components")
		pca = PCA(n_components=50).fit_transform(norm_data[genes, :].T)
		
		if self.k is None:
			k = int(np.min([100, n_real_cells * 0.01]))
		else:
			k = self.k

		logging.info(f" DoubletFinder: Initializing NN structure with k = {k}")
		knn_result = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
		knn_result.fit(pca)
		knn_dist, knn_idx = knn_result.kneighbors(X=pca, return_distance=True)

		knn_result1 = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
		knn_result1.fit(pca[0:n_real_cells, :])
		knn_dist1, _ = knn_result1.kneighbors(X=pca[n_real_cells + 1:, :], n_neighbors=10)
		knn_dist_rc, knn_idx_rc = knn_result1.kneighbors(X=pca[0:n_real_cells, :], return_distance=True)

		logging.info(f" DoubletFinder: Finding the doublet score threshold")
		dist_th = np.mean(knn_dist1.flatten()) + 1.64 * np.std(knn_dist1.flatten())

		doublet_freq = np.logical_and(knn_idx > n_real_cells, knn_dist < dist_th)
		doublet_freq_A = doublet_freq[n_real_cells:n_real_cells + n_doublets, :]
		mean1 = doublet_freq_A.mean(axis=1)
		mean2 = doublet_freq_A[:, 0:int(np.ceil(k / 2))].mean(axis=1)
		doublet_score_A = np.maximum(mean1, mean2)
		
		doublet_freq = doublet_freq[0:n_real_cells, :]
		mean1 = doublet_freq.mean(axis=1)
		mean2 = doublet_freq[:, 0:int(np.ceil(k / 2))].mean(axis=1)
		doublet_score = np.maximum(mean1, mean2)
		doublet_flag = np.zeros(n_real_cells, int)
		doublet_th1 = 1.0
		doublet_th2 = 1.0
		doublet_th = 1.0
		
		# instantiate and fit the KDE model
		kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
		kde.fit(doublet_score_A[:, None])

		# score_samples returns the log of the probability density
		xx = np.linspace(doublet_score_A.min(), doublet_score_A.max(), len(doublet_score_A)).reshape(-1, 1)

		logprob = kde.score_samples(xx)
		if self.fixed_threshold is not None:
			doublet_th = float(self.fixed_threshold)
		else:
			# Check if the distribution is bimodal
			intervals = UniDip(np.exp(logprob)).run()
			if (len(intervals) > 1):
				kmeans = KMeans(n_clusters=2).fit(doublet_score_A.reshape(len(doublet_score_A), 1))
				high_cluster = np.where(kmeans.cluster_centers_ == max(kmeans.cluster_centers_))[0][0]
				doublet_th1 = np.around(np.min(doublet_score_A[kmeans.labels_ == high_cluster]), decimals=3)
			
			# 0.5% for every 1000 cells - the rate of detectable doublets by 10X
			doublet_th2 = np.percentile(doublet_score, 100 - (5e-4 * n_real_cells))
			doublet_th2 = np.around(doublet_th2, decimals=3)
			# The threshold shouldn't be higher than indicated
			if doublet_th2 > self.max_threshold:
				doublet_th2 = self.max_threshold
			if doublet_th1 > self.max_threshold:
				doublet_th1 = self.max_threshold
			if (len(np.where(doublet_score >= doublet_th1)[0]) > (len(np.where(doublet_score >= doublet_th2)[0]))):
				doublet_th = doublet_th2
			else:
				doublet_th = doublet_th1
		logging.info(f" DoubletFinder: Optimal threshold was {doublet_th:.2f}")
		doublet_flag[doublet_score >= doublet_th] = 1

		logging.info(f" DoubletFinder: Finding doublet neighbors")
		# Calculate the score for the cells that are nn of the marked doublets
		if (doublet_flag == 1).sum() > 0:
			pca_rc = pca[0:n_real_cells, :]
			knn_dist1_rc, _ = knn_result1.kneighbors(X=pca_rc[doublet_flag == 1, :], n_neighbors=10, return_distance=True)

			dist_th = np.mean(knn_dist1_rc.flatten()) + 1.64 * np.std(knn_dist1_rc.flatten())
			doublet2_freq = np.logical_and(doublet_flag[knn_idx_rc] == 1, knn_dist_rc < dist_th)
			doublet2_nn = knn_dist_rc < dist_th
			doublet2_score = doublet2_freq.sum(axis=1) / doublet2_nn.sum(axis=1)
			
			doublet_flag[np.logical_and(doublet_flag == 0, doublet2_score >= doublet_th / 2)] = 2
			
		logging.info(f" DoubletFinder: Doublet fraction was {100*len(np.where(doublet_flag > 0)[0]) / n_real_cells:.2f}%, i.e. {len(np.where(doublet_flag > 0)[0])} cells")
		
		return doublet_score, doublet_flag
