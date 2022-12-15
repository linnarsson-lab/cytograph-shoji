from typing import List
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external.pp._scanorama_integrate as si
from cytograph import creates, requires, Algorithm
from sklearn.preprocessing import LabelEncoder
import shoji
import logging


class Scanorama(Algorithm):
    """
    Remove batch effects and intergate
    """
    def __init__(self, batch_keys: List[str] = [], **kwargs) -> None:
        """
        Remove batch effects using Scanorama (Hie et al.2019)
        Args:
            batch_keys: List of names of tensors that indicate batches (e.g. `["Chemistry"]`)
        Writes the output as Factors unless specified creates in the punchcard
        """
        super().__init__(**kwargs)
        self.batch_keys = batch_keys

    @requires("Factors", "float32", ("cells", None))
    #@requires("SelectedFeatures", "bool", ("genes",))
    @creates("Factors", "float32", ("cells",None))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
        if self.batch_keys is not None and len(self.batch_keys) > 0:
            adata = ws.create_anndata(only_selected= True)
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
                    logging.info(f" Scanorama: Can't find {self.batch_keys}")
                    
            adata.obs['batch_ind']= le.fit_transform(adata.obs['batch_concat'].values)
            sc.pp.normalize_total(adata, inplace=True)
            sc.pp.log1p(adata)
            logging.info(f" Scanorama: Integration based on batch keys {batch}")
            si.scanorama_integrate(adata, key = 'batch_ind',basis = 'Factors')
            return adata['X_scanorama']
        else:
            logging.info(" Scanorama: Skipping because no batch keys were provided")

