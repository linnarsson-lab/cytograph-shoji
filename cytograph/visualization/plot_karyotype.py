from cytograph import Module, requires
from .scatter import scatterc, scattern
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import shoji


class PlotKaryotype(Module):
	def __init__(self, filename: str = "karyotype.png", **kwargs) -> None:
		super().__init__(**kwargs)
		self.filename = filename

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("Clusters", "uint32", ("cells",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("Embedding", "float32", ("cells", 2))
	@requires("Gene", "string", ("genes",))
	@requires("Aneuploid", "bool", ("clusters",))
	@requires("KaryotypePloidy", "float32", ("clusters", "karyotype",))
	@requires("KaryotypePosition", "float32", ("karyotype",))
	@requires("KaryotypeChromosome", "string", ("karyotype",))
	@requires("ChromosomeStart", "uint32", ("chromosomes",))
	@requires("ChromosomeLength", "uint32", ("chromosomes",))
	@requires("ChromosomeName", "string", ("chromosomes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
		xy = self.Embedding[:]
		
		plt.figure(figsize=(24, 24))
		plt.subplot(441)
		scatterc(xy, c=self.Clusters[:], s=5, legend=None)
		plt.subplot(442)
		scattern(xy, c=self.Expression[:, self.Gene == "PTPRC"].flatten(), s=5)
		plt.title("PTPRC (immune)")
		plt.subplot(443)
		scattern(xy, c=self.Expression[:, self.Gene == "SOX10"].flatten(), s=5)
		plt.title("SOX10 (oligos)")
		plt.subplot(444)
		scattern(xy, c=self.Expression[:, self.Gene == "EGFR"].flatten(), s=5)
		plt.title("EGFR")
		plt.subplot(445)
		
		scattern(xy, c=self.Aneuploid[:][self.Clusters[:]], s=5)
		plt.title("Aneuploidy (P < 0.05)")
		plt.subplot(446)
		scattern(xy, c=self.Expression[:, self.Gene == "AQP4"].flatten(), s=5)
		plt.title("AQP4 (astros)")
		plt.subplot(447)
		scattern(xy, c=self.Expression[:, self.Gene == "SOX2"].flatten(), s=5)
		plt.title("SOX2 (progenitors)")
		plt.subplot(448)
		scattern(xy, c=self.Expression[:, self.Gene == "DCN"].flatten(), s=5)
		plt.title("DCN (fibroblasts)")

		bottom = plt.cm.get_cmap('Oranges', 128)
		middle = np.full((50, 4), 0.99)
		middle[:, 3] = 1
		top = plt.cm.get_cmap('Blues_r', 128)

		newcolors = np.vstack((top(np.linspace(0, 1, 128)), middle, bottom(np.linspace(0, 1, 128))))
		cmp = ListedColormap(newcolors, name='OrangeBlue')

		ploidy = self.KaryotypePloidy[:]
		chromosomes = self.KaryotypeChromosome[:]

		n_clusters = ploidy.shape[0]

		plt.subplot(212)
		ticks = []
		for ch, start, length in zip(self.ChromosomeName[:], self.ChromosomeStart[:], self.ChromosomeLength[:]):
			plt.imshow(ploidy[:, chromosomes == ch], vmin=0, vmax=4, aspect="auto", cmap=cmp, interpolation="none", extent=(start, start + length, 0, n_clusters), origin="lower")
			plt.vlines(start + length, 0, n_clusters, linestyles="--", color="black", lw=0.5)
			ticks.append(start + length / 2)
		plt.xticks(ticks=ticks, labels=self.ChromosomeName[:])
		plt.ylabel("Cluster")
		plt.xlabel("Chromosome")
		plt.title("Karyotype by cluster")
		plt.colorbar(shrink=0.25, label="Ploidy")

		if save:
			plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
			plt.close()
