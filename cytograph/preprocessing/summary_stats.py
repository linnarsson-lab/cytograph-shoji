from typing import Tuple
import shoji
import numpy as np
import cytograph as cg
from cytograph import requires, creates
import logging


class DetectSpecies:
	def __init__(self) -> None:
		pass

	@requires("Gene", "string", ("genes",))
	@creates("Species", "string", ())
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> str:
		genes = ws[:].Gene
		for gene, species in {
			"ACTB": "Homo sapiens",
			"Tspy1": "Rattus norvegicus",
			"Actb": "Mus musculus",  # Note must come after rat, because rat has the same gene name
			"actb1": "Danio rerio",
			"Act5C": "Drosophila melanogaster",
			"ACT1": "Saccharomyces cerevisiae",
			"act1": "Schizosaccharomyces pombe",
			"act-1": "Caenorhabditis elegans",
			"ACT12": "Arabidopsis thaliana",
			"AFTTAS": "Gallus gallus"
		}.items():
			if gene in genes:
				return species
		raise ValueError("Could not auto-detect species")


class CellSummaryStatistics:
	def __init__(self) -> None:
		pass

	@requires("Expression", None, ("cells", "genes"))
	@requires("Chromosome", "string", ("genes",))
	@requires("Gene", "string", ("genes",))
	@requires("Unspliced", "uint16", ("cells", "genes"))
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
			n_genes				Number of non-zero genes per cell
			n_UMIs				Total UMIs per cell
			mt_fraction			Fraction mitochondrial UMIs per cell
			unspliced_fraction	Fraction unspliced UMIs per cell
			cellcycle_fraction	Fraction cell cycle UMIs per cell
		
		Remarks:
			The complete Expression and Unspliced tensors are loaded into memory
			If species is None, cell cycle scores are set to 0
		"""

		logging.info("SummaryStatistics: Loading 'Expression' and 'Unspliced' tensors")
		x = ws[:].Expression
		u = ws[:].Unspliced

		mito_genes = ws[:].Chromosome == "MT"

		logging.info("CellSummaryStatistics: Computing summary statistics for cells")
		nnz = np.count_nonzero(x, axis=1)
		n_UMIs = np.sum(x, axis=1)
		mt_ratio = np.sum(x[:, mito_genes], axis=1) / n_UMIs
		unspliced_ratio = np.sum(u, axis=1) / n_UMIs

		species = cg.Species(ws[:].Species)
		if species.name in ["Homo sapiens", "Mus musculus"]:
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


class GeneSummaryStatistics:
	def __init__(self) -> None:
		pass

	@requires("Expression", None, ("cells", "genes"))
	@creates("MeanExpression", "float32", ("genes",))
	@creates("StdevExpression", "float32", ("genes",))
	@creates("Nonzeros", "uint32", ("genes",))
	@creates("ValidGenes", "bool", ("genes",))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Calculate summary statistics for each gene

		Args:
			ws				shoji workspace
			save			if true, save the result to the workspace

		Returns:
			mean				Mean expression per gene
			sd					Standard deviation of expression per gene
		"""
		logging.info("GeneSummaryStatistics: Computing summary statistics for genes")
		stats = ws.cells.groupby(None).stats("Expression")
		mu = stats.mean()[1].flatten()
		sd = stats.sd()[1].flatten()
		nnz = stats.nnz()[1].flatten()
		logging.info(f"GeneSummaryStatistics: Done.")
		return (mu, sd, nnz, (nnz > 10) & (nnz < ws.cells.length * 0.6))
