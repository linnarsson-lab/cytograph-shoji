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
import squidpy as sq
import scanpy as sc
hv.notebook_extension('bokeh')

def indices_to_order_a_like_b(a, b):
	return a.argsort()[b.argsort().argsort()]

class PlotSankey(Algorithm):
	def __init__(self, 
				filename: str = "sankey", 
				condense=True, 
				reduce_classes=False, 
				cmap=None,
				cluster_name: str = "Cluster",
				**kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename
		self.condense = condense
		self.reduce_classes = reduce_classes
		self.cmap = cmap
		self.cluster_name = cluster_name

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("MolecularNgh", "int8", ("cells",))
	@requires("Sample", "string", ("cells",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	@requires("AnnotationDescription", "string", ("annotations",))
	@requires("AnnotationName", "string", ("annotations",))
	@requires("AnnotationPosterior", "float32", ("clusters","annotations"))
	@requires("X", "float32", ("cells",))
	@requires("Y", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotSankey: Plotting the graph")

		save_to = self.export_dir / 'Spatial_{}'.format(ws._name )
		if path.exists(save_to) == False:
			makedirs(save_to)

		labels = []
		if self.cluster_name == 'Cluster':
			clusters = self.Clusters[:]#[subsample]
			ClusterID = self.ClusterID[:]
		
		elif self.cluster_name == 'MolecularNgh':
			clusters = self.MolecularNgh[:]
			ClusterID = np.unique(clusters)

		n_clusters = clusters.max() + 1
		x,y = self.X[:], self.Y[:]
		ordering = indices_to_order_a_like_b(ClusterID, np.arange(n_clusters))
		sample = self.Sample[:]#[subsample]

		ann_names = self.AnnotationName[:]
		ann_post = self.AnnotationPosterior[:].T[:, ordering]
		for i in range(ws.clusters.length):
			order = ann_post[:,i].argsort()[::-1]
			label = ann_names[order[:3]]
			label = str(i)+ ' - ' + ' | '.join(label)
			if self.reduce_classes:
				label = ann_names[order[0]]
			labels.append(label)

		unique_samples = np.unique(sample)
		centroids = np.array([x,y]).T
		edges = []
		center_nodes = np.arange(sample.shape[0])
		for s in unique_samples:
			tree = KDTree(centroids[sample==s])
			dst, nghs= tree.query(centroids[sample==s], distance_upper_bound=25, k=10,workers=-1)
			nghs = nghs[:,1:]
			nodes = center_nodes[sample ==s]

			for n,i in zip(nodes, range(nodes.shape[0])):
				for nn in nghs[i,:]:
					try:
						edges.append((n,nodes[nn]))
					except:
						pass
		edges= np.array(edges)

		clustersSource = np.array(labels)[self.Clusters[:][edges[:,0]]]
		clustersTarget = np.array(labels)[self.Clusters[:][edges[:,1]]]
		clustersTarget = np.array( ['Target: '+ t for t in clustersTarget])
		values = np.ones_like(clustersSource)
		df = pd.DataFrame({"source":clustersSource, "target":clustersTarget, "value":values.astype(np.float32)})

		bootstrap = []
		nsim = 10
		for i in range(nsim):
			bootstrapClusters = np.random.choice(clusters,size=clusters.shape[0],replace=True)
			clustersTargetBootstrap = np.array(labels)[bootstrapClusters[edges[:,0]]]
			clustersTargetBootstrap = np.array( ['Target: '+ t for t in clustersTargetBootstrap])
			dfBootstrap = pd.DataFrame({"source":clustersSource, "target":clustersTargetBootstrap, "value":values.astype(np.float32)/nsim})
			bootstrap.append(dfBootstrap)

		df2 = df.pivot_table(
			index='source', 
			columns='target',
			aggfunc='sum',
			fill_value=0,
			dropna=False
		)
		dfBootstrap = pd.concat(bootstrap)
		df2Bootstrap = dfBootstrap.pivot_table(
				index='source', 
				columns='target',
				aggfunc='sum',
				fill_value=1,
				dropna=False
			)

		df2 = df2#/(df2Bootstrap+1)

		df2.columns = [x[1] for x in df2.columns]
		df2Bootstrap.columns = [x[1] for x in df2Bootstrap.columns]
		source,target, v = [], [], []
		for s in df2.index:
			median_v = np.quantile(df2.loc[s,:],0.75)
			for t in df2.columns:
				if df2[t][s] > median_v:
					value_ratio = df2[t][s] #/(df2Bootstrap[t][s]+1)
					v.append(value_ratio)
					source.append(s)
					target.append(t)
		df3 = pd.DataFrame({'source':source,'target':target, 'value':v})
		unique_colors = colorize(np.unique(clusters))
		unique_colors_HEX = [rgb2hex(int(color[0]*255),int(color[1]*255),int(color[2]*255)) for color in unique_colors]
		
		cmap = {t:col for t,col in zip(df2.columns,unique_colors_HEX)}
		cmap2 = {t:col for t,col in zip(df2.index,unique_colors_HEX)}
		cmap = {**cmap,**cmap2}

		if self.condense:
			RUN_df = df3
		else: 
			RUN_df = df

		sankey = hv.Sankey(RUN_df, label='Cell2Cell').opts(
			height=2000,
			width=2000,
			label_position='left', 
			edge_color='target', 
			node_color='index', 
			cmap=cmap,
			edge_muted_alpha=0)
		hv.save(sankey, save_to / (ws._name + "_" + self.filename +".png"))
		hv.save(sankey, save_to / (ws._name + "_" + self.filename + ".html"))

		if type(self.cmap) == type(None):
			cmap_chord = {t:col for t,col in zip(df2.index,unique_colors_HEX)}
		else:
			print('cmap')
			cmap_chord = self.cmap

		df3['target2'] =  [x[8:] for x in df3['target']]# [x[8:] for x in df3['target']]
		data = pd.DataFrame({'s':df3['source'].values,'t':df3['target2'].values})
		try:
			hvdata = hv.Dataset(data)
			chord = hv.Chord((df3,hvdata),['source', 'target2'], ['value'])

			chord = chord.select(s=data.s.values.tolist(), selection_mode='nodes')
			chord.opts(
				hv.opts.Chord(
					width=1000,
					height=1000,
					cmap=cmap_chord,
					edge_color=hv.dim('source').str(),
					node_color=hv.dim('t').str(),
					labels='t',
				)
			)
			hv.save(chord, save_to / (ws._name + "_chord.png"),dpi=500)
			hv.save(chord, save_to / (ws._name + "_chord.html"))

			chord_magma = chord.select(s=data.s.values.tolist(), selection_mode='nodes')
			chord_magma.opts(
				hv.opts.Chord(
					width=1000,
					height=1000,
					cmap=cmap_chord,
					edge_cmap='magma',
					edge_color=hv.dim('source').str(),
					node_color=hv.dim('t').str(),
					labels='t',
				)
			)

			hv.save(chord_magma, save_to / (ws._name + "_chord_edges.html"))
		except:
			print('Chord diagram failed')

		graph = hv.Graph(((df3['source'].values,df3['target2'].values, df3['value'].values),),vdims='value').opts(
            hv.opts.Graph(
				edge_cmap='magma',edge_color='value',node_color='index',
				cmap=cmap_chord,edge_line_width=hv.dim('value')*0.005,
                edge_nonselection_alpha=0, width=1500,height=1500
            )
		)

		labels = hv.Labels(graph.nodes, ['x', 'y'],'index')
		graph = graph * labels.opts(text_font_size='8pt', text_color='black', bgcolor='white')
		hv.save(graph, save_to / (ws._name + "_graph.html"))


class PlotNeighborhood(Algorithm):
	def __init__(self, 
				filename: str = "sankey", 
				condense=True, 
				reduce_classes=False, 
				cmap=None,
				cluster_name: str = "Cluster",
				**kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename
		self.condense = condense
		self.reduce_classes = reduce_classes
		self.cmap = cmap
		self.cluster_name = cluster_name

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("MolecularNgh", "int8", ("cells",))
	@requires("Sample", "string", ("cells",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	@requires("AnnotationDescription", "string", ("annotations",))
	@requires("AnnotationName", "string", ("annotations",))
	@requires("AnnotationPosterior", "float32", ("clusters","annotations"))
	@requires("X", "float32", ("cells",))
	@requires("Y", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotSankey: Plotting the graph")
		#subsample = np.random.choice(np.arange(self.Embedding[:].shape[0]),size=50000,replace=False)
		save_to = self.export_dir / 'Spatial_{}'.format(ws._name )
		if path.exists(save_to) == False:
			makedirs(save_to)

		labels = []
		if self.cluster_name == 'Cluster':
			clusters = self.Clusters[:]#[subsample]
			ClusterID = self.ClusterID[:]
		
		elif self.cluster_name == 'MolecularNgh':
			clusters = self.MolecularNgh[:]
			ClusterID = np.unique(clusters)

		samples = self.Sample[:]
		unique_samples = np.unique(samples)
		X = self.X[:]
		Y = self.Y[:]
		expression = self.Expression[:]
		for sample in unique_samples:
			save_to = self.export_dir / 'Spatial_{}_{}'.format(ws._name , sample)
			if path.exists(save_to) == False:
				makedirs(save_to)
				
			filter_sample = samples == sample

			leiden = pd.Categorical(clusters[filter_sample])

			spatial = np.array([X[filter_sample], Y[filter_sample]]).T
			adata = sc.AnnData(
				X= expression[filter_sample,:],
				obsm={'spatial':spatial}, 
				obs={'cell type':leiden}
			)

			'''sq.gr.co_occurrence(adata, cluster_key="cell type", interval=50)
			for cell_ in adata.obs['cell type'].cat.categories:
				try:
					sq.pl.co_occurrence(
						adata,
						cluster_key="cell type",
						clusters=[cell_],
						figsize=(10, 10),
						palette='Set1',
						save = save_to / (ws._name + "_co-ocurrence{}.png".format(cell_) ),
					)
				except:
					logging.info('Co-ocurrence plot failed fro cluster {}'.format(cell_))'''

			sq.gr.spatial_neighbors(
				adata, 
				#delaunay=True,
				coord_type='generic',
				radius=50,
				)

			sq.gr.nhood_enrichment(
				adata, 
				cluster_key="cell type")

			sq.pl.nhood_enrichment(
				adata, 
				cluster_key="cell type", 
				cmap='magma',
				mode='count',
				method='ward',
				save=save_to / (ws._name + "_neighborhood.png"))
