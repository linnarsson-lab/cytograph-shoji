from typing import List
import numpy as np
import scanpy as sc
import cytograph as cg
from cytograph import requires, creates, Algorithm
import shoji
import logging


class FeatureSelectionSeuratV3(Algorithm):
    """
    Select features by Seurat V3 method
    """
    def __init__(self, n_genes: int, mask: List[str] = None, batch: List[str] = None, **kwargs) -> None:
        """
        Use scanpy function to select high-variance genes using seurat V3 method (Stuart et al. (2019))

        Args:
            n_genes		Number of genes to select
            mask		Optional mask (numpy bool array) indicating genes that should not be selected
            batch       Optional batch/s to consider
        Remarks:
            If the tensor "ValidGenes" exists, only ValidGenes == True genes will be selected
        """
        super().__init__(**kwargs)
        self.n_genes = n_genes
        self.mask = mask if mask is not None else []
        self.batch = batch if batch is not None else []

    @requires("Species", "string", ())
    @requires("Expression", None, ("cells", "genes"))
    @creates("SelectedFeatures", "bool", ("genes",), indices=True)
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
        n_genes = ws.genes.length
        species = cg.Species(self.Species[:])
        mask_genes = species.mask(ws, self.mask)

        logging.info(" FeatureSelectionSeuratV3: Initiating ")


        if "ValidGenes" in ws:
            valid = ws.ValidGenes[:]
        else:
            valid = np.ones(n_genes, dtype='bool')
        if self.mask is not None:
            valid = np.logical_and(valid, np.logical_not(mask_genes))

        #Get adata
        adata = ws.create_anndata(only_selected= False)
        sc.pp.normalize_total(adata, target_sum=1e4, layer="Expression")
        sc.pp.log1p(adata, layer="Expression")
        if(len(self.batch)>0):
            for i in range(0,len(self.batch)):
                if(i ==0):
                    adata.obs["batch"] = adata.obs[self.batch[i]].astype("str")
                else:
                    adata.obs["batch"] =  adata.obs["batch"]+"-"+adata.obs[self.batch[i]].astype("str")

            adata_v = adata[:,valid].copy()
            sc.pp.highly_variable_genes(
            adata_v,
            flavor="seurat_v3",
            n_top_genes=self.n_genes,
            layer="Expression",
            batch_key="batch",
            subset=False)
        else:
            adata_v = adata[:,valid].copy()
            sc.pp.highly_variable_genes(
            adata_v,
            flavor="seurat_v3",
            n_top_genes=self.n_genes,
            layer="counts",
            subset=False)
        
        genes = np.where(valid)[0][adata_v.var['highly_variable']]
        logging.info(f" FeatureSelectionSeuratV3: Selected {len(genes)} features")
        return genes
