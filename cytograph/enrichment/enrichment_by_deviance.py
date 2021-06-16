import shoji
import numpy as np
from cytograph import requires, creates, Module
from ..utils import div0
import logging


class EnrichmentByDeviance(Module):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	@requires("GeneTotalUMIs", "uint32", ("genes",))
	@requires("MeanExpression", None, ("clusters", "genes"))
	@creates("Enrichment", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		"""
		Calculate binomial deviance statistics for each gene and cluster

		Remarks:
			See equation 9 on p. 4 of https://doi.org/10.1101/2020.12.01.405886
		"""
		logging.info(" EnrichmentByDeviance: Computing the Pearson residuals (deviance) per cluster")
		data = self.MeanExpression[:].astype("float32")
		totals = data.sum(axis=1)
		gene_totals = self.GeneTotalUMIs[:].astype("float32")

		expected = totals[:, None] @ div0(gene_totals[None, :], totals.sum())
		residuals = div0((data - expected), np.sqrt(expected + np.power(expected, 2) / 100))

		return residuals
