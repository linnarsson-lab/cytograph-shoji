import matplotlib.patheffects as PathEffects
from matplotlib.collections import LineCollection
import logging

import matplotlib.pyplot as plt
import numpy as np

import shoji
from cytograph import Algorithm, requires
from .scatter import scatterc
from scipy.spatial import KDTree


class PlotManifold(Algorithm):
	def __init__(self, filename: str = "manifold.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotManifold: Plotting the embedding")

		labels = []
		n_cells = self.NCells[:]
		clusters = self.Clusters[:]
		n_clusters = clusters.max() + 1
		cluster_ids = self.ClusterID[:]
		genes = self.Gene[:]
		xy = self.Embedding[:]
		ordering = np.argsort(ws.ClusterID[:])
		mean_x = self.MeanExpression[:][ordering]
		enrichment = self.Enrichment[:][ordering]
		idx_zeros = np.where(mean_x.sum(axis=0) == 0)[0]
		enrichment[:, idx_zeros] = np.zeros([enrichment.shape[0],idx_zeros.shape[0]])

		for i in range(ws.clusters.length):
			n = n_cells[cluster_ids == i][0]
			label = f"{i:>3} ({n:,} cells) "
			label += " ".join(genes[np.argsort(-enrichment[cluster_ids == i, :][0])[:10]])
			labels.append(label)

		plt.figure(figsize=(20, 20))
		ax = plt.subplot(111)
		'''if "ManifoldIndices" in ws:
			edges = ws.ManifoldIndices[:]
			lc = LineCollection(zip(xy[edges[:, 0]], xy[edges[:, 1]]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
			ax.add_collection(lc)'''

		MAX_CLUSTERS = 100
		top_clusters = np.argsort(np.bincount(clusters))[-MAX_CLUSTERS:]
		for i in top_clusters:
			pos = np.median(xy[clusters == i], axis=0)
			txt = plt.text(pos[0], pos[1], str(i), size=18, color="black")
			txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

		if n_clusters > MAX_CLUSTERS:
			mask = np.isin(clusters, top_clusters)
			clusters[~mask] = n_clusters
			labels.append(f"{n_clusters} ({n_clusters - MAX_CLUSTERS} clusters not shown)")
		scatterc(xy, c=np.array(labels)[clusters], legend="outside")
		plt.axis("off")
		logging.info("export dir: "+str(self.export_dir))
		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
		plt.close()


class PlotManifoldGraph(Algorithm):
	def __init__(self, filename: str = "manifoldgraph.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("GraphCluster", "int8", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotManifold: Plotting the embedding")

		labels = []
		n_cells = self.NCells[:]
		clusters = self.Clusters[:]
		graphclusters = self.GraphCluster[:]
		n_clusters = graphclusters.max() + 1
		cluster_ids = self.ClusterID[:]
		genes = self.Gene[:]
		enrichment = self.Enrichment[:]
		xy = self.Embedding[:]

		for gi in range(n_clusters):
			if (graphclusters == gi).sum() > 50:
				#i = np.unique(clusters[graphclusters == gi])
				n = (graphclusters ==gi).sum()
				label = f" {gi:>3} ({n:,} cells) - "
				labels.append(label)
			else:
				graphclusters[graphclusters == gi] = -1
				labels.append("-1 (0 cells) ")

		plt.figure(figsize=(20, 20))
		ax = plt.subplot(111)
		if "ManifoldIndices" in ws:
			edges = ws.ManifoldIndices[:]
			lc = LineCollection(zip(xy[edges[:, 0]], xy[edges[:, 1]]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
			ax.add_collection(lc)

		MAX_CLUSTERS = 100
		top_clusters = np.argsort(np.unique(graphclusters,return_counts=True)[1])[-MAX_CLUSTERS:]
		for i in top_clusters:
			pos = np.median(xy[graphclusters == i], axis=0)
			txt = plt.text(pos[0], pos[1], str(i), size=18, color="black")
			txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

		if n_clusters > MAX_CLUSTERS:
			mask = np.isin(graphclusters, top_clusters)
			graphclusters[~mask] = n_clusters
			labels.append(f"{n_clusters} ({n_clusters - MAX_CLUSTERS} clusters not shown)")
		
		scatterc(xy, c=np.array(labels)[graphclusters], legend="outside")
		plt.axis("off")
		logging.info("export dir: "+str(self.export_dir))
		plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
		plt.close()


'''class PlotManifoldNeighborsGraph(Algorithm):
	def __init__(self, filename: str = "manifold.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	@requires("X", "float32", ("cells",))
	@requires("Y", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:

		

		sample = ws.Sample[:]
		x,y = ws.X[:], ws.Y[:]
		global_cell = np.arange(x.shape[0])
		neighbor_cells = []
		unique_samples = np.unique(sample)
		centroids = np.array([x,y]).T
		edges = []
		center_nodes = np.arange(sample.shape[0])
		for s in unique_samples:
			print(s)
			tree = KDTree(centroids[sample==s])
			dst, nghs= tree.query(centroids[sample==s], distance_upper_bound=75, k=4,workers=-1)
			nghs = nghs[:,1:]
			nodes = center_nodes[sample ==s]

			for n,i in zip(nodes, range(nodes.shape[0])):
				for nn in nghs[i,:]:
					try:
						edges.append((n,nodes[nn]))
					except:
						pass
		edges= np.array(edges)
		ws.NeighborsIndices = shoji.Tensor("uint64", (edges.shape[0],edges.shape[1]), inits=edges.astype(np.uint64))
		logging.info(" PlotManifold: Plotting the embedding")
		
		labels = []
		n_cells = ws.NCells[:]
		clusters = ws.Clusters[:]
		n_clusters = clusters.max() + 1
		cluster_ids = ws.ClusterID[:]
		genes = ws.Gene[:]
		enrichment = ws.Enrichment[:]
		xy = ws.Embedding[:]

		for i in range(ws.clusters.length):
			n = n_cells[cluster_ids == i][0]
			label = f"{i:>3} ({n:,} cells) "
			label += " ".join(genes[np.argsort(-enrichment[cluster_ids == i, :][0])[:10]])
			labels.append(label)

		plt.figure(figsize=(20, 20))
		ax = plt.subplot(111)
		if "NeighborsIndices" in ws:
			edges = ws.NeighborsIndices[:]
			lc = LineCollection(zip(xy[edges[:, 0]], xy[edges[:, 1]]), linewidths=0.02, zorder=0, color='black', alpha=0.1)
			ax.add_collection(lc)

		MAX_CLUSTERS = 100
		top_clusters = np.argsort(np.bincount(clusters))[-MAX_CLUSTERS:]
		for i in top_clusters:
			pos = np.median(xy[clusters == i], axis=0)
			txt = plt.text(pos[0], pos[1], str(i), size=18, color="black")
			txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

		if n_clusters > MAX_CLUSTERS:
			mask = np.isin(clusters, top_clusters)
			clusters[~mask] = n_clusters
			labels.append(f"{n_clusters} ({n_clusters - MAX_CLUSTERS} clusters not shown)")
		scatterc(xy, c=np.array(labels)[clusters], legend="outside")
		plt.axis("off")
		plt.savefig(ws.export_dir / (ws._name + "_" + ws.filename+"Neighbors"), dpi=500, bbox_inches='tight')
		plt.close()'''
