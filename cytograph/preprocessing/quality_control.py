from typing import Tuple, Union
import shoji
import numpy as np
from cytograph import requires, creates, Algorithm
import logging


class QualityControl(Algorithm):
	"""
	Compute QC metrics and mark valid cells (note: consider using ClassifyDroplets instead)
	"""
	def __init__(self, doublet_threshold: Union[float, str] = "auto", min_umis: int = 0, max_mt_fraction: float = 1, min_unspliced_fraction: float = 0, min_fraction_good_cells: float = 0, **kwargs) -> None:
		"""
		Args:
			doublet_threshold		Threshold to call doublets, or "auto" to use automatic threshold (default: "auto")
			min_umis				Minimum number of UMIs (default: 0)
			max_mt_fraction			Maximum fraction mitochondrial UMIs (default: 1)
			min_unspliced_fraction	Minimum fraction unspliced UMIs (default: 0)
			min_fraction_good_cells	Minimum fraction of cells that must pass QC for the sample to pass as a whole (default: 0)
		"""
		super().__init__(**kwargs)
		self.doublet_threshold = doublet_threshold
		self.min_umis = min_umis
		self.max_mt_fraction = max_mt_fraction
		self.min_unspliced_fraction = min_unspliced_fraction
		self.min_fraction_good_cells = min_fraction_good_cells

	@requires("MitoFraction", "float32", ("cells",))
	@requires("UnsplicedFraction", "float32", ("cells",))
	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("DoubletScore", "float32", ("cells",))
	@requires("DoubletFlag", "bool", ("cells",))
	@creates("ValidCells", "bool", ("cells",))
	@creates("PassedQC", "bool", ())
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Find cells that pass quality control criteria

		Args:
			ws				shoji workspace
			save			if true, save the result to the workspace

		Returns:
			ValidCells		Bool tensor indicating cells that passed QC criteria
			PassedQC		Scalar indicating if the sample as a whole passed QC
		
		Remarks:
			The complete Expression and Unspliced tensors are loaded into memory
			If species is None, cell cycle scores are set to 0
		"""
		n_cells = ws.cells.length
		if self.doublet_threshold == "auto":
			good_cells = ~self.DoubletFlag[:]
		else:
			good_cells = self.DoubletScore[:] < self.doublet_threshold
		logging.info(f" QualityControl: Marked {n_cells - good_cells.sum()} doublets")

		enough_umis = self.TotalUMIs[:] >= self.min_umis
		good_cells &= enough_umis
		logging.info(f" QualityControl: Marked {n_cells - enough_umis.sum()} cells with too few UMIs")

		good_mito_fraction = self.MitoFraction[:] < self.max_mt_fraction
		good_cells &= good_mito_fraction
		logging.info(f" QualityControl: Marked {n_cells - good_mito_fraction.sum()} cells with too high mitochondrial UMI fraction")

		good_unspliced_fraction = self.UnsplicedFraction[:] >= self.min_unspliced_fraction
		good_cells &= good_unspliced_fraction
		logging.info(f" QualityControl: Marked {n_cells - good_unspliced_fraction.sum()} cells with too low unspliced UMI fraction")

		passed_qc = True
		good_fraction = good_cells.sum() / n_cells
		if good_fraction < self.min_fraction_good_cells:
			logging.warning(f" QualityControl: Sample failed QC because only {good_cells.sum()} of {n_cells} cells passed")
			passed_qc = False
		else:
			logging.info(f" QualityControl: {good_cells.sum()} of {n_cells} cells ({int(100 * good_fraction)}%) passed QC")
		return (good_cells, np.array(passed_qc, dtype=bool))

class QualityControlEEL(Algorithm):
	"""
	Compute QC metrics and mark valid cells (note: consider using ClassifyDroplets instead)
	"""
	def __init__(self, min_umis: int = 10, min_genes:int=3, min_fraction_good_cells: float = 0, 
		umi_gene_ratio: float = 1.2, remove_graphclusters=[], **kwargs) -> None:
		"""
		Args:
			doublet_threshold		Threshold to call doublets, or "auto" to use automatic threshold (default: "auto")
			min_umis				Minimum number of UMIs (default: 0)
			max_mt_fraction			Maximum fraction mitochondrial UMIs (default: 1)
			min_unspliced_fraction	Minimum fraction unspliced UMIs (default: 0)
			min_fraction_good_cells	Minimum fraction of cells that must pass QC for the sample to pass as a whole (default: 0)
		"""
		super().__init__(**kwargs)
		self.min_umis = min_umis
		self.min_genes = min_genes
		self.min_fraction_good_cells = min_fraction_good_cells
		self.umi_gene_ratio = umi_gene_ratio
		self.remove_graphclusters = remove_graphclusters

	@requires("TotalUMIs", "uint32", ("cells",))
	@requires("GraphCluster", "int8", ("cells",))
	@creates("ValidCells", "bool", ("cells",))
	@creates("PassedQC", "bool", ())
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Find cells that pass quality control criteria

		Args:
			ws				shoji workspace
			save			if true, save the result to the workspace

		Returns:
			ValidCells		Bool tensor indicating cells that passed QC criteria
			PassedQC		Scalar indicating if the sample as a whole passed QC
		
		Remarks:
			The complete Expression and Unspliced tensors are loaded into memory
			If species is None, cell cycle scores are set to 0
		"""
		
		n_cells = ws.cells.length
		enough_umis = self.TotalUMIs[:] >= self.min_umis
		enough_genes = (ws.Expression[:,:] > 0).sum(axis=1) >= self.min_genes
		umis_gene_ratio = self.TotalUMIs[:] / enough_genes
		enough_ratio = umis_gene_ratio > self.umi_gene_ratio
		keep_graph = np.isin(self.GraphCluster[:], self.remove_graphclusters, invert=True)
		good_cells = enough_umis & enough_genes & enough_ratio & keep_graph

		logging.info(f" QualityControl: Marked {n_cells - enough_umis.sum()} cells with too few UMIs")

		passed_qc = True
		good_fraction = good_cells.sum() / n_cells
		if good_fraction < self.min_fraction_good_cells:
			logging.warning(f" QualityControl: Sample failed QC because only {good_cells.sum()} of {n_cells} cells passed")
			passed_qc = False
		else:
			logging.info(f" QualityControl: {good_cells.sum()} of {n_cells} cells ({int(100 * good_fraction)}%) passed QC")
		return (good_cells, np.array(passed_qc, dtype=bool))

