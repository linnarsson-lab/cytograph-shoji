from typing import Tuple
import shoji
import numpy as np
import cytograph as cg
from cytograph import requires, creates, Algorithm
import logging


class GeneSummaryStatistics(Algorithm):
	def __init__(self, **kwargs) -> None:
		"""
		Calculate summary statistics for each gene

		Creates:
			MeanExpression		Mean expression per gene
			StdevExpression		Standard deviation of expression per gene
			Nonzeros			Nonzero count per gene
			GeneTotalUMIs		Total UMIs per gene
			ValidGenes			Bool array indicating genes with nnz > 10
		"""
		super().__init__(**kwargs)

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("TotalUMIs", "uint32", ("cells",))
	@creates("MeanExpression", "float32", ("genes",))
	@creates("StdevExpression", "float32", ("genes",))
	@creates("Nonzeros", "uint32", ("genes",))
	@creates("GeneTotalUMIs", "uint32", ("genes",))
	@creates("ValidGenes", "bool", ("genes",))
	@creates("OverallTotalUMIs", "uint64", ())
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		logging.info(" GeneSummaryStatistics: Computing summary statistics for genes")
		stats = ws.cells.groupby(None).stats(self.requires["Expression"])
		mu = stats.mean()[1].flatten()
		sd = stats.sd()[1].flatten()
		nnz = stats.nnz()[1].flatten()
		s = stats.sum()[1].flatten()
		valids = (nnz > 10)
		logging.info(f" GeneSummaryStatistics: Average nonzero cells per gene {int(nnz.mean())}")
		logging.info(f" GeneSummaryStatistics: Average UMIs per gene {int(s.mean())}")
		logging.info(f" GeneSummaryStatistics: Number of valid genes {valids.sum():,} ({int(valids.sum() / ws.genes.length * 100)}%)")
		total = self.TotalUMIs[:].sum()
		logging.info(f" GeneSummaryStatistics: Total number of UMIs in dataset {total:,}")
		return (mu, sd, nnz, s, valids, total)
