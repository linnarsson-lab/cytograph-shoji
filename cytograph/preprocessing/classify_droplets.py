from typing import Tuple
import numpy as np
from scipy.stats import multivariate_normal
import shoji
from cytograph import requires, creates, Module
import logging


class GaussianMLE(object):
	# http://www.sciencedirect.com/science/article/pii/0024379585900497
	def __init__(self, X):
		# X is a numpy array of size (observations) x (variables)
		self.X = X
		self.N = X.shape[1]
		self.M = X.shape[0]
		self.fit = False

		# estimate the sample mean for each variable
		mean = np.mean(self.X, axis=0, dtype=float)
		self.mu = mean
		# estimate the variance covariance matrix
		diffs = self.X - self.mu
		dot = np.dot(diffs.T, diffs)
		sigma = dot / self.M
		self.sigma = sigma

	def pdf(self, x):
		return multivariate_normal.pdf(x, self.mu, self.sigma)


class ClassifyDroplets(Module):
	"""
	Classify droplets based on total UMI count and unspliced fraction
	"""
	def __init__(self, min_umis: int = 1500, min_unspliced_fraction: float = 0.1, m: int = 200, k: int = 1000, min_pdf: float = 0.1, max_doublet_score: float = 0.4, max_mito_fraction: float = 0.25, **kwargs) -> None:
		"""
		Args:
			min_umis                 Minimum UMIs for valid cells
			min_unspliced_fraction   Minimum unspliced fraction for valid cells
			m                        UMI intercept for valid cells
			k                        UMI per unspliced fraction slope for valid cells
			min_pdf                  Minimum gaussian pdf for valid cells
			max_doublet_score        Max allowed doublet score for valid cells
			max_mito_fraction        Max allowed mitochondrial UMI fraction for valid cells
		"""
		super().__init__(**kwargs)
		self.min_umis = min_umis
		self.min_unspliced_fraction = min_unspliced_fraction
		self.m = m
		self.k = k
		self.min_pdf = min_pdf
		self.max_doublet_score = max_doublet_score
		self.max_mito_fraction = max_mito_fraction
	
	@requires("UnsplicedFraction", "float32", ("cells",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("DoubletScore", "float32", ("cells",))
	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("Chromosome", "string", ("genes",))
	@creates("DropletClass", "uint8", ("cells",))
	@creates("ValidCells", "bool", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		logging.info(" ClassifyDroplets: Fitting Gaussian MLE")

		classes = np.zeros(ws.cells.length, dtype="uint8")
		# 0 = passed, 1 = large cells, 2 = doublets, 3 = cytoplasmic debris, 4 = cellular debris, 5 = nuclear debris, 6 = mitochondrial debris
		
		u, t = self.UnsplicedFraction[:], np.log10(self.TotalUMIs[:])
		selected = ((t > np.log10(self.m) + np.log10(self.k) * u) & (u > self.min_unspliced_fraction) & (t > np.log10(self.min_umis)))
		X_selected = np.vstack([u[selected], t[selected]]).T
		X_all = np.vstack([u, t]).T

		X_selected_scaled = X_selected.copy()
		X_selected_scaled[:, 0] = X_selected_scaled[:, 0] * 2

		X_all_scaled = X_all.copy()
		X_all_scaled[:, 0] = X_all_scaled[:, 0] * 2

		gaussian = GaussianMLE(X_selected_scaled)
		passed = gaussian.pdf(X_all_scaled) > self.min_pdf
		cellular_max_unspliced = np.percentile(X_all[passed, 0], 99)
		cellular_min_unspliced = np.percentile(X_all[passed, 0], 1)

		classes[~passed] = 5  # Default is nuclear debris, but we'll refine this below
		classes[~passed & (X_all[:, 0] < cellular_max_unspliced)] = 4
		classes[~passed & (X_all[:, 0] < cellular_min_unspliced)] = 3
		
		total_UMIs = 10 ** t
		mt_fraction = self.Expression[:, self.Chromosome == "MT"].sum(axis=1) / total_UMIs
		if mt_fraction.sum() == 0:
			mt_fraction = self.Expression[:, self.Chromosome == "chrM"].sum(axis=1) / total_UMIs
		if mt_fraction.sum() == 0:
			mt_fraction = self.Expression[:, self.Chromosome == "M"].sum(axis=1) / total_UMIs
		classes[mt_fraction > self.max_mito_fraction] = 6
		classes[selected & ~passed & (total_UMIs > np.median(total_UMIs))] = 1
		classes[self.DoubletScore[:] > self.max_doublet_score] = 2
		logging.info(f" ClassifyDroplets: {int((classes == 0).sum() / classes.shape[0])}% cells passed")
		return classes, classes == 0
