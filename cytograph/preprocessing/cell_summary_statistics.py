from typing import Tuple
import shoji
import numpy as np
import cytograph as cg
from cytograph import requires, creates, Module
import logging


class CellSummaryStatistics(Module):
	def __init__(self, **kwargs) -> None:
		super().__init__(**kwargs)

	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("Chromosome", "string", ("genes",))
	@requires("Gene", "string", ("genes",))
	@requires("Unspliced", None, ("cells", "genes"))
	@requires("Species", "string", ())
	@creates("NGenes", "uint32", ("cells",))
	@creates("TotalUMIs", "uint32", ("cells",))
	@creates("MitoFraction", "float32", ("cells",))
	@creates("UnsplicedFraction", "float32", ("cells",))
	@creates("CellCycleFraction", "float32", ("cells",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Calculate summary statistics for each cell

		Args:
			ws				shoji workspace
			save			if true, save the result to the workspace

		Returns:
			NGenes				Number of non-zero genes per cell
			TotalUMIs			Total UMIs per cell
			MitoFraction		Fraction mitochondrial UMIs per cell
			UnsplicedFraction	Fraction unspliced UMIs per cell
			CellCycleFraction	Fraction cell cycle UMIs per cell
		
		Remarks:
			The complete Expression and Unspliced tensors are loaded into memory
			If species is None, cell cycle scores are set to 0
		"""
		logging.info(" CellSummaryStatistics: Loading 'Expression' and 'Unspliced' tensors")
		x = self.Expression[:]
		u = self.Unspliced[:].astype("uint16")

		mito_genes = self.Chromosome[:] == "MT"

		logging.info(f" CellSummaryStatistics: Computing summary statistics for {ws.cells.length} cells")
		nnz = np.count_nonzero(x, axis=1)
		n_UMIs = np.sum(x, axis=1)
		mt_ratio = np.sum(x[:, mito_genes], axis=1) / n_UMIs
		unspliced_ratio = np.sum(u, axis=1) / n_UMIs

		species = cg.Species(self.Species[:].item())
		if species.name in ["Homo sapiens", "Mus musculus"]:
			genes = self.Gene[:]
			g1_indices = np.isin(genes, species.genes.g1)
			s_indices = np.isin(genes, species.genes.s)
			g2m_indices = np.isin(genes, species.genes.g2m)
			g1 = x[:, g1_indices].sum(axis=1)
			s = x[:, s_indices].sum(axis=1)
			g2m = x[:, g2m_indices].sum(axis=1)
			cc = (g1 + s + g2m) / n_UMIs
		else:
			cc = np.zeros(nnz.shape[0], dtype="float32")

		logging.info(f" CellSummaryStatistics: Average number of non-zero genes {int(nnz.mean())}")
		logging.info(f" CellSummaryStatistics: Average total UMIs {int(n_UMIs.mean())}")
		logging.info(f" CellSummaryStatistics: Average mitochondrial UMI fraction {100*mt_ratio.mean():.2f}%")
		logging.info(f" CellSummaryStatistics: Average unspliced fraction {100*unspliced_ratio.mean():.2f}%")
		if species.name not in ["Homo sapiens", "Mus musculus"]:
			logging.info(f" CellSummaryStatistics: Average cell cycle UMI fraction was not calculated because species not given")
		else:
			logging.info(f" CellSummaryStatistics: Average cell cycle UMI fraction {100*cc.mean():.2f}%")
			logging.info(f" CellSummaryStatistics: Fraction of cycling cells {100 * (cc > 0.01).sum() / cc.shape[0]:.2f}%")
		return (nnz, n_UMIs, mt_ratio, unspliced_ratio, cc)
