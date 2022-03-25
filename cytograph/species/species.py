from types import SimpleNamespace
from typing import Any, List

import numpy as np

import shoji
from cytograph import requires, creates

from .human import TFs_human, cc_genes_human, g1_human, g2m_human, s_human
from .mouse import TFs_mouse, cc_genes_mouse, g1_mouse, g2m_mouse, s_mouse


class Species:
	def __init__(self, name) -> None:
		self.name = name
		if self.name == "Homo sapiens":
			genes = {
				"TFs": TFs_human,
				"cellcycle": cc_genes_human,
				"sex": ["XIST", "TSIX"],
				"ieg": ['JUNB', 'FOS', 'EGR1', 'JUN'],
				"g1": g1_human,
				"s": s_human,
				"g2m": g2m_human,
				"ery": ["HBA-A1", "HBA-A2", "HBA-X", "HBB-BH1", "HBB-BS", "HBB-BT", "HBB-Y", "ALAS2"],
				"mt": ['MT-CYB', 'MT-ND6', 'MT-CO3', 'MT-ND1', 'MT-ND4', 'MT-CO1', 'MT-ND2', 'MT-CO2', 'MT-ATP8', 'MT-ND4L', 'MT-ATP6', 'MT-ND5', 'MT-ND3']
			}
			self.markers = {
				"CellCycle": ["PCNA", "CDK1", "TOP2A"],
				"RadialGlia": ["HES1", "HOPX"],
				"Immune": ["PTPRC"],
				"Macrophages": ["AIF1", "HEXB", "MRC1"],
				"Fibroblasts": ["LUM", "DCN", "COL1A1"],
				"Endothelial": ["CLDN5"],
				"Erythrocyte": ["HBB"],
				"VSMC": ["ACTA2", "TAGLN"],
				"Ependymal": ["TMEM212", "FOXJ1"],
				"Choroid": ["TTR"],
				"Astrocytes": ["AQP4", "GJA1"],
				"Neurons": ["INA"],
				"Neuroblasts": ["NHLH1", "NHLH2"],
				"GABAergic": ["GAD1", "GAD2", "SLC32A1"],
				"Glycinergic": ["SLC6A5", "SLC6A9"],
				"Excitatory": ["SLC17A7", "SLC17A8", "SLC17A6"],
				"Serotonergic": ["TPH2", "FEV"],
				"Dopaminergic": ["TH", "SLC6A3"],
				"Cholinergic": ["CHAT", "SLC5A7", "SLC18A3"],
				"Monoamine": ["SLC18A1", "SLC18A2"],
				"Noradrenergic": ["DBH"],
				"Adrenergic": ["PNMT"],
				"Oligodendrocytes": ["PDGFRA", "PLP1", "SOX10", "MOG", "MBP"],
				"Schwann": ["MPZ"],
				"Sex": ["XIST", "TSIX", "SRY"]
			}
		elif self.name == "Mus musculus":
			genes = {
				"TFs": TFs_mouse,
				"cellcycle": cc_genes_mouse,
				"sex": ["Xist", "Tsix"],
				"ieg": ['Junb', 'Fos', 'Egr1', 'Jun'],
				"g1": g1_mouse,
				"s": s_mouse,
				"g2m": g2m_mouse,
				"ery": ["Hba-a1", "Hba-a2", "Hba-x", "Hbb-bh1", "Hbb-bs", "Hbb-bt", "Hbb-y", "Alas2"],
				"mt": ['mt-Nd1', 'mt-Nd2', 'mt-Co1', 'mt-Co2', 'mt-Atp8', 'mt-Atp6', 'mt-Co3', 'mt-Nd3', 'mt-Nd4l', 'mt-Nd4', 'mt-Nd5', 'mt-Cytb', 'mt-Nd6']
			}
			self.markers = {
				"CellCycle": ["Pcna", "Cdk1", "Top2a"],
				"RadialGlia": ["Fabp7", "Fabp5", "Hopx"],
				"Immune": ["PTPRC"],
				"Macrophages": ["Aif1", "Hexb", "Mrc1"], 
				"Fibroblasts": ["Lum", "Dcn", "Col1a1"], 
				"Endothelial": ["Cldn5"], 
				"VSMC": ["Acta2", "Tagln"],
				"Ependymal": ["Tmem212", "Foxj1"],
				"Astrocytes": ["Aqp4", "Gja1"],
				"Neurons": ["Rbfox3"],
				"Neuroblasts": ["Nhlh1", "Nhlh2"],
				"GABAergic": ["Gad1", "Gad2", "Slc32a1"], 
				"Glycinergic": ["Slc6a5", "Slc6a9"],
				"Excitatory": ["Slc17a7", "Slc17a8", "Slc17a6"],
				"Serotonergic": ["Tph2", "Fev"],
				"Dopaminergic": ["Th", "Slc6a3"],
				"Cholinergic": ["Chat", "Slc5a7", "Slc18a3"],
				"Monoamine": ["Slc18a1", "Slc18a2"],
				"Noradrenergic": ["Dbh"],
				"Adrenergic": ["Pnmt"],
				"Oligodendrocytes": ["Pdgfra", "Plp1", "Sox10", "Mog", "Mbp"],
				"Schwann": ["Mpz"]
			}
		else:
			genes = {
				"TFs": [],
				"cellcycle": [],
				"sex": [],
				"ieg": [],
				"g1": [],
				"s": [],
				"g2m": [],
				"ery": [],
				"mt": []
			}

			self.markers = {}
		self.genes = SimpleNamespace(**genes)

	def mask(self, ws: shoji.WorkspaceManager, categories: List[str]) -> np.ndarray:
		"""
		Create a boolean mask that includes all genes except those that belong to any of the categories

		Args:
			categories		Any combination of "TFs", "cellcycle", "sex", "ieg", "g1", "s", "g2m"
		"""
		mask = np.zeros(ws.genes.length, dtype=bool)
		genes = ws[:].Gene
		for cat in categories:
			mask = mask | np.isin(genes, self.genes.__dict__[cat])
		return mask
