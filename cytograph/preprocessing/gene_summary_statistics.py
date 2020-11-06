from typing import Tuple
import shoji
import numpy as np
import cytograph as cg
from cytograph import requires, creates
import logging


class GeneSummaryStatistics:
	def __init__(self) -> None:
		pass

	@requires("Expression", "uint16", ("cells", "genes"))
	@creates("MeanExpression", "float32", ("genes",))
	@creates("StdevExpression", "float32", ("genes",))
	@creates("Nonzeros", "uint32", ("genes",))
	@creates("GeneTotalUMIs", "uint32", ("genes",))
	@creates("ValidGenes", "bool", ("genes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Calculate summary statistics for each gene

		Args:
			ws				shoji workspace
			save			if true, save the result to the workspace

		Returns:
			MeanExpression		Mean expression per gene
			StdevExpression		Standard deviation of expression per gene
			Nonzeros			Nonzero count per gene
			GeneTotalUMIs		Total UMIs per gene
			ValidGenes			Bool array indicating genes with nnz > 10 and fraction nonzeros less than 60%
		"""
		logging.info(" GeneSummaryStatistics: Computing summary statistics for genes")
		stats = ws.cells.groupby(None).stats("Expression")
		mu = stats.mean()[1].flatten()
		sd = stats.sd()[1].flatten()
		nnz = stats.nnz()[1].flatten()
		s = stats.sum()[1].flatten()
		valids = (nnz > 10) & (nnz < ws.cells.length * 0.6)
		logging.info(f" GeneSummaryStatistics: Average nonzero cells per gene {int(nnz.mean())}")
		logging.info(f" GeneSummaryStatistics: Average UMIs per gene {int(s.mean())}")
		logging.info(f" GeneSummaryStatistics: Number of valid genes {valids.sum()} ({int(valids.sum() / ws.genes.length * 100)}%)")
		return (mu, sd, nnz, s, valids)
