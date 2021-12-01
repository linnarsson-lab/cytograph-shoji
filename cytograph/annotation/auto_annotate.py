import logging
import os
import re
from typing import List, Tuple

from cytograph import creates, requires, Module

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
	def __init__(self, path: str, **kwargs) -> None:
		"""
		Args:
			path: 	Path to the auto-annotation database
		"""
		super().__init__(**kwargs)
		self.root = path

	@requires("Gene", "string", ("genes",))
	@requires("Trinaries", "float32", ("clusters", "genes"))
	@creates("AnnotationName", "string", ("annotations",))
	@creates("AnnotationDefinition", "string", ("annotations",))
	@creates("AnnotationDescription", "string", ("annotations",))
	@creates("AnnotationPosterior", "float32", ("clusters", "annotations"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Compute auto-annotation for the workspace using the given definitions
		"""
		genes = self.Gene[:]
		n_clusters = ws.clusters.length

		pp = self.Trinaries[:]
		definitions: List[Annotation] = []

		fileext = [".yaml", ".md"]
		root_len = len(self.root)
		for cur, _, files in os.walk(self.root):
			for file in files:
				if os.path.splitext(file)[-1] in fileext and file[-9:] != "README.md":
					tag = Annotation(cur[root_len:], os.path.join(cur, file))
					definitions.append(tag)

		def annotation_posterior(positives, negatives):
			posteriors = np.ones(n_clusters)
			for gene in positives:
				posteriors *= pp[:, genes == gene].flatten()[:n_clusters]  # Take only the first n_clusters in case the gene name is duplicated
			for gene in negatives:
				posteriors *= (1 - pp[:, genes == gene].flatten())[:n_clusters]
			return posteriors

		posteriors = np.empty((len(definitions), n_clusters))
		for ix, tag in enumerate(definitions):
			for gene in tag.positives + tag.negatives:
				if gene not in genes:
					logging.error(f"Gene '{gene}' for tag '{tag.name}' not found in the workspace and will omitted from the tag definition.")
					tag.definition += f" (!missing '{gene}')"
					continue
			posteriors[ix, :] = annotation_posterior(tag.positives, tag.negatives)

		# Recreate the annotations dimension
		if save:
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

