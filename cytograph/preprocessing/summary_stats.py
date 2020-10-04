from typing import Tuple
import shoji
import numpy as np
from cytograph.species import Species
from cytograph import CytographMethod
import logging


class CellSummaryStatistics(CytographMethod):
	def __init__(self, species: Species) -> None:
		"""
		Args:
			species		A cytograph Species object, which is used to obtain the list of cell cycle genes
		"""
		self.species = species
		self._requires = [
			("Expression", None, ("cells", "genes")),
			("Unspliced", "uint16", ("cells", "genes")),
			("Chromosome", "string", ("genes",)),
			("Gene", "string", ("genes",))
		]

	def fit(self, ws: shoji.WorkspaceManager) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Calculate summary statistics for each cell

		Args:
			ws				shoji workspace

		Returns:
			n_genes				Number of non-zero genes per cell
			n_UMIs				Total UMIs per cell
			mt_fraction			Fraction mitochondrial UMIs per cell
			unspliced_fraction	Fraction unspliced UMIs per cell
			cellcycle_fraction	Fraction cell cycle UMIs per cell
		
		Remarks:
			The complete Expression and Unspliced tensors are loaded into memory
			If species is None, cell cycle scores are set to 0
		"""
		self.check(ws, "CellSummaryStatistics")

		logging.info("SummaryStatistics: Loading 'Expression' and 'Unspliced' tensors")
		x = ws[:].Expression
		u = ws[:].Unspliced

		mito_genes = ws[:].Chromosome == "MT"

		logging.info("CellSummaryStatistics: Computing summary statistics for cells")
		nnz = np.count_nonzero(x, axis=1)
		n_UMIs = np.sum(x, axis=1)
		mt_ratio = np.sum(x[:, mito_genes], axis=1) / n_UMIs
		unspliced_ratio = np.sum(u, axis=1) / n_UMIs

		if self.species is not None:
			genes = ws[:].Gene
			g1_indices = np.isin(genes, self.species.genes.g1)
			s_indices = np.isin(genes, self.species.genes.s)
			g2m_indices = np.isin(genes, self.species.genes.g2m)
			g1 = x[:, g1_indices].sum(axis=1)
			s = x[:, s_indices].sum(axis=1)
			g2m = x[:, g2m_indices].sum(axis=1)
			cc = (g1 + s + g2m) / n_UMIs
		else:
			cc = 0

		logging.info(f"CellSummaryStatistics: Average number of non-zero genes {int(nnz.mean())}")
		logging.info(f"CellSummaryStatistics: Average total UMIs {int(n_UMIs.mean())}")
		logging.info(f"CellSummaryStatistics: Average mitochondrial UMI fraction {100*mt_ratio.mean():.2f}%")
		logging.info(f"CellSummaryStatistics: Average unspliced fraction {100*unspliced_ratio.mean():.2f}%")
		if self.species is None:
			logging.info(f"CellSummaryStatistics: Average cell cycle UMI fraction was not calculated because species not given")
		else:
			logging.info(f"CellSummaryStatistics: Average cell cycle UMI fraction {100*cc.mean():.2f}%")
		logging.info(f"CellSummaryStatistics: Done.")
		return (nnz, n_UMIs, mt_ratio, unspliced_ratio, cc)

	def fit_save(self, ws: shoji.WorkspaceManager) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		(nnz, n_UMIs, mt_ratio, unspliced_ratio, cc) = self.fit(ws)
		logging.info("CellSummaryStatistics: Saving nnz as uint32 tensor 'NGenes'")
		logging.info("CellSummaryStatistics: Saving n_UMIs as uint32 tensor 'TotalUMIs'")
		logging.info("CellSummaryStatistics: Saving mitochondrial UMI fraction as float32 tensor 'MitoFraction'")
		logging.info("CellSummaryStatistics: Saving unspliced fraction as float32 tensor 'UnsplicedFraction'")
		logging.info("CellSummaryStatistics: Saving cell cycle UMI fraction as float32 tensor 'CellCycleFraction'")
		logging.info("CellSummaryStatistics: Saving mean expression per gene as float32 tensor 'MeanExpression'")
		logging.info("CellSummaryStatistics: Saving standard deviation per gene as float32 tensor 'StdevExpression'")
		ws.NGenes = shoji.Tensor("uint32", ("cells",), inits=nnz.astype("uint32"))
		ws.TotalUMIs = shoji.Tensor("uint32", ("cells",), inits=n_UMIs.astype("uint32"))
		ws.MitoFraction = shoji.Tensor("float32", ("cells",), inits=mt_ratio.astype("float32"))
		ws.UnsplicedFraction = shoji.Tensor("float32", ("cells",), inits=unspliced_ratio.astype("float32"))
		ws.CellCycleFraction = shoji.Tensor("float32", ("cells",), inits=cc.astype("float32"))
		return (nnz, n_UMIs, mt_ratio, unspliced_ratio, cc)


class GeneSummaryStatistics(CytographMethod):
	def __init__(self) -> None:
		self._requires = [
			("Expression", None, ("cells", "genes"))
		]

	def fit(self, ws: shoji.WorkspaceManager) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Calculate summary statistics for each gene

		Args:
			ws				shoji workspace

		Returns:
			mean				Mean expression per gene
			sd					Standard deviation of expression per gene
		"""
		self.check(ws, "GeneSummaryStatistics")
		logging.info("GeneSummaryStatistics: Computing summary statistics for genes")
		stats = ws.cells.groupby(None).stats("Expression")
		mu = stats.mean()[1].flatten()
		sd = stats.sd()[1].flatten()
		logging.info(f"GeneSummaryStatistics: Done.")
		return (mu, sd)

	def fit_save(self, ws: shoji.WorkspaceManager) -> Tuple[np.ndarray, np.ndarray]:
		(mu, sd) = self.fit(ws)
		logging.info("GeneSummaryStatistics: Saving mean expression per gene as float32 tensor 'MeanExpression'")
		logging.info("GeneSummaryStatistics: Saving standard deviation per gene as float32 tensor 'StdevExpression'")
		ws.MeanExpression = shoji.Tensor("float32", ("genes",), inits=mu.astype("float32"))
		ws.StdevExpression = shoji.Tensor("float32", ("genes",), inits=sd.astype("float32"))
		return (mu, sd)
