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
from colormap import rgb2hex
from scipy.spatial import KDTree
import holoviews as hv
hv.notebook_extension('bokeh')

def indices_to_order_a_like_b(a, b):
	return a.argsort()[b.argsort().argsort()]

class PlotSankey(Algorithm):
	def __init__(self, filename: str = "sankey", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Gene", "string", ("genes",))
	@requires("Clusters", "uint32", ("cells",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("NCells", "uint64", ("clusters",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Sample", "string", ("cells",))
	@requires("Enrichment", "float32", ("clusters", "genes"))
	@requires("AnnotationDescription", "string", ("annotations",))
	@requires("AnnotationName", "string", ("annotations",))
	@requires("AnnotationPosterior", "float32", ("clusters","annotations"))
	@requires("X", "float32", ("cells",))
	@requires("Y", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		logging.info(" PlotSankey: Plotting the graph")
		subsample = np.random.choice(np.arange(self.Embedding[:].shape[0]),size=50000,replace=False)
		labels = []
		clusters = self.Clusters[:]#[subsample]
		n_clusters = clusters.max() + 1
		x,y = self.X[:], self.Y[:]
		ordering = indices_to_order_a_like_b(self.ClusterID[:], np.arange(n_clusters))
		sample = self.Sample[:]#[subsample]

		ann_names = self.AnnotationName[:]
		ann_post = self.AnnotationPosterior[:].T[:, ordering]
		for i in range(ws.clusters.length):
			order = ann_post[:,i].argsort()[::-1]
			label = ann_names[order[:3]]
			label = str(i)+ ' - ' + ' | '.join(label)
			labels.append(label)

		unique_samples = np.unique(sample)
		centroids = np.array([x,y]).T
		edges = []
		center_nodes = np.arange(sample.shape[0])
		for s in unique_samples:
			print(s)
			tree = KDTree(centroids[sample==s])
			dst, nghs= tree.query(centroids[sample==s], distance_upper_bound=75, k=3,workers=-1)
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

		df2 = df.pivot_table(
		index='source', 
		columns='target',
		aggfunc='sum',
		fill_value=0,
		dropna=False
		)
		df2.columns = [x[1] for x in df2.columns]
		source,target, v = [], [], []
		for s in df2.index:
			median_v = df2.loc[s,:].median()
			print(median_v)
			for t in df2.columns:
				if df2[t][s] > median_v:
					v.append(df2[t][s])
					source.append(s)
					target.append(t)
		df3 = pd.DataFrame({'source':source,'target':target, 'value':v})
		unique_colors = colorize(np.unique(clusters))
		unique_colors_HEX = [rgb2hex(int(color[0]*255),int(color[1]*255),int(color[2]*255)) for color in unique_colors]
		cmap = {t:col for t,col in zip(df2.columns,unique_colors_HEX)}
		cmap2 = {t:col for t,col in zip(df2.index,unique_colors_HEX)}
		cmap = {**cmap,**cmap2}

		sankey = hv.Sankey(df3, label='Cell2Cell').opts(
			height=2000,
			width=1000,
			label_position='left', 
			edge_color='target', 
			node_color='index', 
			cmap=cmap,
			edge_muted_alpha=0)
		hv.save(sankey, self.export_dir / (ws._name + "_" + self.filename +".png"))
		hv.save(sankey, self.export_dir / (ws._name + "_" + self.filename + ".html"))


		cmap_chord = {t:col for t,col in zip(df2.index,unique_colors_HEX)}
		df3['target2'] =  [x[8:] for x in df3['target']]
		data = pd.DataFrame({'s':df2.index.values,'t':df2.index.values})

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

		hv.save(sankey, self.export_dir / (ws._name + "_chord.png"))
		hv.save(sankey, self.export_dir / (ws._name + "_chord.html"))
