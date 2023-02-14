
import shoji
import numpy as np
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from cytograph import Algorithm, requires, creates
import scipy
import logging
import pandas as pd

class SmoothNN(Algorithm):
	def __init__(self, max_distance=50, **kwargs) -> None:
		super().__init__(**kwargs)
		self.max_distance = max_distance


	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("MolecularNgh", "int8", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("Sample", "string", ("cells",))
	@requires("X", "float32", ("cells",))
	@requires("Y", "float32", ("cells",))
	@creates("SmoothExpression", "uint16", ("cells", "genes"))
	@creates("MolecularNghMerge", "int8", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		clusters = self.MolecularNgh[:]
		ClusterID = np.unique(clusters)
		expression = self.Expression[:]
		gene =self.Gene[:]
		n_clusters = clusters.max() + 1
		x,y = self.X[:], self.Y[:]
		sample = self.Sample[:]#[subsample]
		unique_samples = np.unique(sample)
		centroids = np.array([x,y]).T
		hm = []
		for c in ClusterID:
				hm.append(expression[clusters ==c,:].mean(axis=0))
		hm = np.stack(hm)

		hm = pd.DataFrame(StandardScaler().fit_transform(hm).T, columns=ClusterID, index=gene)
		Z = scipy.cluster.hierarchy.linkage(hm.T, method='average', metric='correlation')
		merged_labels_short = scipy.cluster.hierarchy.fcluster(Z, 0.25, criterion='distance')
		logging.info(f"Reduced {n_clusters} clusters to {merged_labels_short.max()} clusters")
		dic_merged_clusters = dict(zip(ClusterID, merged_labels_short))
		clusters = np.array([dic_merged_clusters[c] for c in clusters])

		expression_smooth = []
		for s in unique_samples:
			clusters_s = clusters[sample==s]
			centroids_sample = centroids[sample==s]
			tree = KDTree(centroids_sample)
			dst, nghs= tree.query(centroids_sample, distance_upper_bound=self.max_distance, k=25,workers=-1,p=2)
			nghs = [n[n < clusters_s.shape[0]] for n in nghs]
			clusters_nghs= [clusters[n] for n in nghs]
			nghs= [n[c == c[0]]  for c, n in zip(clusters_nghs, nghs)]
			expression_ = np.array([expression[n,:].sum(axis=0) for n in nghs])
			expression_smooth.append(expression_)

		expression_smooth = np.concatenate(expression_smooth)
		return expression_smooth, clusters


