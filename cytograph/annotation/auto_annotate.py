import logging
import os
import re
from typing import List, Tuple

from numpy.core.fromnumeric import size
from cytograph import creates, requires, Module
from cytograph.pipeline import Config
import numpy_groupies as npg
from scipy.special import gammaincc

import numpy as np
import yaml

import shoji


class Annotation:
	unknown_tags: set = set()

	def __init__(self, category: str, filename: str) -> None:
		with open(filename) as f:
			doc = next(yaml.load_all(f, Loader=yaml.SafeLoader))

		if "name" in doc:
			self.name = doc["name"]
		else:
			raise ValueError(os.path.basename(filename) + " did not contain a 'name' attribute, which is required.")

		if "abbreviation" in doc:
			self.abbreviation = doc["abbreviation"]
		else:
			raise ValueError(os.path.basename(filename) + " did not contain an 'abbreviation' attribute, which is required.")

		if "definition" in doc:
			self.definition = doc["definition"]
			genes = self.definition.strip().split()
			self.positives = [x[1:] for x in genes if x.startswith("+")]
			self.negatives = [x[1:] for x in genes if x.startswith("-")]
		else:
			raise ValueError(os.path.basename(filename) + " did not contain a 'definition' attribute, which is required.")

		if "categories" in doc and doc["categories"] is not None:
			self.categories = re.split(r"\W+", doc["categories"].strip())
		else:
			self.categories = []

	def __str__(self) -> str:
		temp = self.name + " (" + self.abbreviation + "; " + " ".join(["+" + x for x in self.positives])
		if len(self.negatives) > 0:
			temp = temp + " " + " ".join(["-" + x for x in self.negatives]) + ")"
		else:
			temp = temp + ")"
		return temp


class AutoAnnotate(Module):
	def __init__(self, path: str, threshold: float = 1, **kwargs) -> None:
		"""
		Args:
			path: 		Path to the auto-annotation database
			threshold: 	Expression threshold required (default: 1)
		"""
		super().__init__(**kwargs)
		self.root = path
		self.threshold = threshold

	@requires("Gene", "string", ("genes",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("OverallTotalUMIs", "uint64", ())
	@requires("Clusters", "uint32", ("cells",))
	@requires("NCells", "uint64", ("clusters",))
	@requires("MeanExpression", None, ("clusters", "genes"))
	@creates("AnnotationName", "string", ("annotations",))
	@creates("AnnotationDefinition", "string", ("annotations",))
	@creates("AnnotationDescription", "string", ("annotations",))
	@creates("AnnotationPosterior", "float32", ("clusters", "annotations"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Compute auto-annotation for the workspace using the given definitions
			
		Remarks
			Gamma-Poisson estimate of expression level
			Assume that counts are Poisson distributed
			Use a non-informative prior Gamma(0, 0)
			Given n observations with total sum x, the posterior distribution is Gamma(x, n)
			Instead of counting n = 1 for each cell, the count is adjusted relative to the cell size (total UMI count). 
			The cumulative density function is used to find the posterior probability that expression exceeds a desired threshold λ:
				p = 1 - (gammaincc(x, 0) - gammaincc(x, λ * n))
		"""
		genes = self.Gene[:]
		n_clusters = ws.clusters.length
		n_cells_per_cluster = self.NCells[:]
		totals = self.TotalUMIs[:]
		labels = self.Clusters[:]

		# Calculate the posterior probabilities that expression exceeds the threshold
		# Total molecules detected in each cluster
		x = (self.MeanExpression[:].T * n_cells_per_cluster).T + 1  # Add one pseudocount to avoid nan
		# Relative size of each cell
		size_factors = totals / np.median(totals)
		# Total adjusted size of each cluster (number of median-sized cells)
		n = npg.aggregate(labels, size_factors, func="sum")
		# Posterior probability that a gene is expressed above the threshold with high confidence
		pp = 1 - (gammaincc(x, 0) - gammaincc(x, self.threshold * n[:, None]))
		definitions: List[Annotation] = []
	
		fileext = [".yaml", ".md"]
		root_len = len(self.root)
		for cur, _, files in os.walk(self.root):
			for file in files:
				errors = False
				if os.path.splitext(file)[-1] in fileext and file[-9:] != "README.md":
					try:
						tag = Annotation(cur[root_len:], os.path.join(cur, file))
						for pos in tag.positives:
							if pos not in genes:
								logging.error(file + ": gene '%s' not found in workspace", pos)
								errors = True
						for neg in tag.negatives:
							if neg not in genes:
								logging.error(file + ": gene '%s' not found in workspace", neg)
								errors = True
						if not errors:
							definitions.append(tag)
					except Exception as e:
						logging.error(file + ": " + str(e))
						errors = True

		posteriors = np.empty((len(definitions), n_clusters))
		for ix, tag in enumerate(definitions):
			for cluster in range(n_clusters):
				p = 1
				for pos in tag.positives:
					p = p * pp[cluster, np.where(pos == genes)[0][0]]
				for neg in tag.negatives:
					p = p * (1 - pp[cluster, np.where(pos == genes)[0][0]])
				posteriors[ix, cluster] = p

		# Recreate the annotations dimension
		if "annotations" in ws._dimensions():
			del ws.AnnotationName
			del ws.AnnotationDefinition
			del ws.AnnotationDescription
			del ws.AnnotationPosterior
		ws.annotations = shoji.Dimension(shape=len(definitions))

		names = np.array([ann.abbreviation for ann in definitions], dtype="object")
		defs = np.array([ann.definition for ann in definitions], dtype="object")
		descs = np.array([str(ann) for ann in definitions], dtype="object")
		return names, defs, descs, posteriors.T
