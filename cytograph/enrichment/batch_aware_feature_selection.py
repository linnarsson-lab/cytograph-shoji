from typing import List
import numpy as np
import cytograph as cg
from cytograph import requires, creates, Algorithm
import shoji
import logging


def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a, b)
		c[~np.isfinite(c)] = 0  # -inf inf NaN
	return c


class BatchAwareFeatureSelection(Algorithm):
	"""
	Batch-aware feature selection using Pearson residuals
	"""
	def __init__(self, n_genes: int, batch_key: str = None, mask: List[str] = None, max_nnz_ratio: float = 4, **kwargs) -> None:
		"""
		Args:
			n_genes:		Number of genes to select
			batch_key:		The name of the tensor that indicates batches
			mask:			Optional list indicating categories of genes that should not be selected
			max_nnz_ratio:  Max ratio allowed between highest and lower non-zero fraction in batches

		Remarks:
			Only ValidGenes == True genes will be selected
		"""
		super().__init__(**kwargs)
		self.n_genes = n_genes
		self.batch_key = batch_key
		self.mask = mask if mask is not None else []
		self.max_nnz_ratio = max_nnz_ratio

	@requires("Species", "string", ())
	@requires("Expression", "uint16", ("cells", "genes"))
	@requires("PearsonResiduals", "float32", (None, "genes"))
	@requires("ValidGenes", "bool", ("genes",))
	@requires("PearsonResidualsVariance", "float32", ("genes",))
	@creates("SelectedFeatures", "bool", ("genes",), indices=True)
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		species = cg.Species(self.Species[:])
		mask_genes = species.mask(ws, self.mask)
		logging.info(f" BatchAwareFeatureSelection: Masking {mask_genes.sum():,} genes in {self.mask}")

		logging.info(" BatchAwareFeatureSelection: Computing non-zero ratios per batch")
		stats = ws.cells.groupby(self.batch_key).stats(self.requires["PearsonResiduals"])
		nnzs = stats.nnz()[1]  # shape (n_batches, n_genes)
		counts = stats.count()[1]
		nnz_fractions = nnzs / counts
		nnz_ratios = div0(nnz_fractions.max(axis=0), nnz_fractions.min(axis=0))
		nnz_masked = (nnz_ratios > self.max_nnz_ratio) | (nnz_ratios < 1 / self.max_nnz_ratio)
		logging.info(f" BatchAwareFeatureSelection: Masking {nnz_masked.sum():,} genes with nnz ratio > {self.max_nnz_ratio}")
		mask_genes |= nnz_masked
		
		# Load variance of residuals (which will be batch-corrected if they were computed using BatchAwarePearsonResiduals)
		valid = self.ValidGenes[:]
		valid = np.logical_and(valid, np.logical_not(mask_genes))
		logging.info(f" BatchAwareFeatureSelection: Considering {(valid).sum():,} valid and unmasked genes")

		d = self.PearsonResidualsVariance[:]
		temp = []
		for gene in np.argsort(-d):
			if valid[gene]:

				temp.append(gene)
			if len(temp) >= self.n_genes:
				break
		genes = np.sort(np.array(temp))
		logging.info(f" BatchAwareFeatureSelection: Selected the top {len(genes)} genes")
		return genes
