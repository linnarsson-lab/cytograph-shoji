from typing import List
import numpy as np
import pandas as pd
import scanpy as sc
from cytograph import creates, requires, Algorithm
from sklearn.preprocessing import LabelEncoder
import shoji
import logging

class SCVI(Algorithm):
    """
    Remove batch effects and intergate
    """
    def __init__(self, n_factors=50, batch_keys: List[str] = [], **kwargs) -> None:
        """
        Remove batch effects using scVI (Lopez et al. 2018)
        Args:
            batch_keys: List of names of tensors that indicate batches (e.g. `["Chemistry"]`)
        Writes the output as Factors unless mentioned otherwise in the punch card
        """
        super().__init__(**kwargs)
        self.n_factors= n_factors
        self.batch_keys = batch_keys

    @requires("Factors", "float32", ("cells", None))
    @requires("SelectedFeatures", "bool", ("genes",))
    @creates("Factors", "float32", ("cells",None))
    
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
            import scanpy.external.pp._scanorama_integrate as si
            import scvi
            import scanpy as sc
            if self.batch_keys is not None and len(self.batch_keys) > 0:
                    adata = sc.AnnData(X=ws.Expression[:],obs={self.batch_keys[0]:ws.Sample[:]})
                    sc.pp.highly_variable_genes(
                        adata,
                        n_top_genes=500,
                        flavor="seurat_v3",
                    )
                    le = LabelEncoder() 
                    batch = self.batch_keys
                    for b in self.batch_keys:
                        if(b in adata.obs):
                            if(type(sc.get.obs_df(adata,keys=b)[0]) is not str):
                                adata.obs[b+'_c'] = adata.obs[b].astype(str)
                                batch.remove(b)
                                batch.append(b+'_c')
                                b = b+'_c'
                            if('batch_concat' in adata.obs):
                                adata.obs['batch_concat'] = np.add(adata.obs['batch_concat'],adata.obs[b])      
                            else:
                                adata.obs['batch_concat'] = adata.obs[b]
                        else:
                            logging.info(f" SCVI: Can't find  {b}")
                    adata.obs['batch_ind']= le.fit_transform(adata.obs['batch_concat'].values)
                    #if(len(np.unique(adata.obs['batch_ind']))>1):
                    logging.info(f" SCVI: Integration based on batch keys {batch}")
                    n_epochs=np.min([round((20000/adata.n_obs)*400), 400])
                    n_hidden=128
                    n_layers=2
                    scvi._settings.ScviConfig.num_threads = 12
                    scvi._settings.ScviConfig.batch_size = 512
                    scvi.model.SCVI.setup_anndata(adata, batch_key='batch_ind')
                    model = scvi.model.SCVI(adata, n_latent=self.n_factors,gene_likelihood='zinb',n_layers=n_layers,n_hidden=n_hidden)
                    model.train()
                    latent = model.get_latent_representation()
                        
                    #else:
                    #    logging.info(f" SCVI: Skipping because one batch was found")
                    #    latent = self.Factors[:]
                        
                
            else:
                logging.info(" scVI: Skipping because no batch keys were provided")               
                latent = self.Factors[:]
            return(latent)    
