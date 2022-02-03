from statsmodels.nonparametric.smoothers_lowess import lowess
from ..utils import div0
from ..module import requires, creates, Module
import numpy as np
import shoji


def windowed_mean(x: np.ndarray, n: int):
	if len(x) == 0:
		return x
	y = np.zeros_like(x)
	for ix in range(len(x)):
		i = min(ix, len(x) - n)
		w = x[i:i + n]
		y[ix] = np.mean(w)
	return y


class Karyotyper(Module):
	"""
	Estimate the karyotype of tumor cells using immune cells as internal reference
	"""
	def __init__(self, max_rv = 3, min_umis = 1, window_size: int = 20, smoothing_bandwidth: int = 0):
		"""
		Estimate the karyotype

		Args:
			max_rv:              Max residual variance for selecting housekeeping genes
			min_umis:            Minimum UMIs for selecting housekeeping genes
			window_size:         Number of genes per window
			smooting_bandwidth:  Optional bandwidth for smoothing

		Remarks:
			This algorithm first identifies immune cells using the M-IMMUNE auto-annotation
			(e.g. PTPRC expression) and computes a reference normal karyotype. It then
			estimates the karyotype of each cluster and calls each as aneuploid or euploid.
		"""
		self.max_rv = max_rv
		self.min_umis = min_umis
		self.window_size = window_size
		self.smoothing_bandwidth = smoothing_bandwidth
		
		# The genome offsets of the starts of all the true chromosomes
		self.chr_starts = {
			'1': 0,
			'2': 248956422,
			'3': 491149951,
			'4': 689445510,
			'5': 879660065,
			'6': 1061198324,
			'7': 1232004303,
			'8': 1391350276,
			'9': 1536488912,
			'10': 1674883629,
			'11': 1808681051,
			'12': 1943767673,
			'13': 2077042982,
			'14': 2191407310,
			'15': 2298451028,
			'16': 2400442217,
			'17': 2490780562,
			'18': 2574038003,
			'19': 2654411288,
			'20': 2713028904,
			'21': 2777473071,
			'22': 2824183054,
			'X': 2875001522,
			'Y': 3031042417
		}

	@requires("PearsonResidualsVariance", "float32", ("genes",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("Chromosome", "string", ("genes",))
	@requires("Start", "int64", ("genes",))
	@requires("ClusterID", "uint32", ("clusters",))
	@requires("MeanExpression", "float64", ("clusters", "genes"))
	@requires("AnnotationName", "string", ("annotations",))
	@requires("AnnotationPosterior", "float32", ("clusters", "annotations"))
	@creates("Aneuploid", "bool", ("clusters",))
	@creates("KaryotypePloidy", "float32", ("clusters", "karyotype",))
	@creates("KaryotypePosition", "float32", ("karyotype",))
	@creates("KaryotypeChromosome", "string", ("karyotype",))
	@creates("ChromosomeStart", "uint32", ("chromosomes",))
	@creates("ChromosomeLength", "uint32", ("chromosomes",))
	@creates("ChromosomeName", "string", ("chromosomes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False):
		# Identify housekeeping genes
		self.housekeeping = (self.PearsonResidualsVariance[:] < self.max_rv) & ((self.GeneTotalUMIs[:] / ws.cells.length) > self.min_umis)

		# Order by genomic position
		chrs = self.Chromosome[self.housekeeping]
		starts = self.Start[self.housekeeping]
		for chrom in self.chr_starts.keys():
			starts[chrs == chrom] += self.chr_starts[chrom]
		self.ordering = np.argsort(starts)
		self.chromosomes = chrs[self.ordering]
		self.starts = starts[self.ordering]

		# Load the aggregated expression data
		y_sample = self.MeanExpression[:]
		totals = y_sample.sum(axis=1)
		y_sample = (y_sample.T / totals * np.median(totals)).T
		y_sample = y_sample[:, self.housekeeping]
		self.y_sample = y_sample[:, self.ordering]
		self.y_sample_mean = self.y_sample.mean(axis=1)

		# Load the immune cell clusters
		assert "M-IMMUNE" in self.AnnotationName[:], "M-IMMUNE auto-annotation is required for Karyotyper"
		self.immune = (self.AnnotationPosterior[:, self.AnnotationName[:] == "M-IMMUNE"] > 0.9)[:, 0]
		y_ref = self.MeanExpression[self.immune, :]
		totals = y_ref.sum(axis=1)
		y_ref = (y_ref.T / totals * np.median(totals)).T
		y_ref = y_ref[:, self.housekeeping]
		self.y_ref = y_ref[:, self.ordering].mean(axis=0)
		self.y_ref_mean = self.y_ref.mean()

		# Bin locally along each chromosome
		for chrom in self.chr_starts.keys():
			selected = (self.chromosomes == chrom)
			if selected.sum() == 0:
				continue
			self.y_ref[selected] = windowed_mean(self.y_ref[selected], self.window_size)
			for i in range(self.y_sample.shape[0]):
				self.y_sample[i, selected] = windowed_mean(self.y_sample[i, selected], self.window_size)
		
		# Center using the reference residuals
		self.y_ratio = (div0(self.y_sample, self.y_ref).T * self.y_ref_mean / self.y_sample_mean).T

		if self.smoothing_bandwidth > 0:
			# Loess smoothing along each chromosome
			self.y_ratio_smooth = np.copy(self.y_ratio)
			for chrom in self.chr_starts.keys():
				selected = (self.chromosomes == chrom)
				for i in range(self.y_ratio_smooth.shape[0]):
					self.y_ratio_smooth[i, selected] = lowess(self.y_ratio_smooth[i, selected], self.starts[selected], frac=min(0.5, self.smoothing_bandwidth / selected.sum()), return_sorted=False)
		else:
			self.y_ratio_smooth = None

		if "karyotype" in ws:
			del ws.KaryotypePloidy
			del ws.KaryotypePosition
			del ws.KaryotypeChromosome
			del ws.Aneuploid
			del ws.karyotype

		ws.karyotype = shoji.Dimension(shape=self.y_ratio.shape[1])
		ws.chromosomes = shoji.Dimension(shape=len(self.chr_starts))

		sig_std = np.percentile(np.std(self.y_ratio[self.immune], axis=1), 95)
		aneuploid = np.std(self.y_ratio, axis=1)[self.ClusterID[:]] > sig_std

		chr_starts = np.array(list(self.chr_starts.values()))
		chr_lengths = np.diff(np.append(chr_starts, int(3.04e9)))
		chr_names = np.array(list(self.chr_starts.keys()))

		return aneuploid, 2 * self.y_ratio, self.starts, self.chromosomes, chr_starts, chr_lengths, chr_names
