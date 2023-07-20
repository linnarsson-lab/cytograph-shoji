from typing import Union, Tuple, Literal
import shoji
import numpy as np
from cytograph import requires, creates, Algorithm
import logging
from pathlib import Path


class ExportH5Ad(Algorithm):
	"""
	Export the workspace as .h5ad file compatible with scanpy and cellxgene
	"""
	def __init__(self,
		X: str = "Expression",
		var: Union[Literal["auto"], Tuple[str]] = "auto",
		obs: Union[Literal["auto"], Tuple[str]] = "auto",
		varm: Union[Literal["auto"], Tuple[str]] = "auto",
		obsm: Union[Literal["auto"], Tuple[str]] = "auto",
		var_key: str = "Accession",
		obs_key: str = "CellID",
		layers: Tuple[str] = (),
		**kwargs
	) -> None:
		super().__init__(**kwargs)
		self.X = X
		self.var = var
		self.obs = obs
		self.varm = varm
		self.obsm = obsm
		self.var_key = var_key
		self.obs_key = obs_key
		self.layers = layers

	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> str:
		logging.info(" ExportH5Ad: Creating AnnData object")
		ad = ws.anndata(X=self.X, var=self.var, obs=self.obs, varm=self.varm, obsm=self.obsm, var_key=self.var_key, obs_key=self.obs_key, layers=self.layers)
		logging.info(" ExportH5Ad: Saving AnnData as .h5ad file")
		ad.write_h5ad(Path(self.export_dir) / (ws._name + ".h5ad"))
