from typing import List

import numpy as np
import pandas as pd
from harmony import harmonize
from cytograph import creates, requires, Module
import shoji
import logging


class Harmony(Module):
	"""
	Remove batch effects
	"""
	def __init__(self, batch_keys: List[str] = [], **kwargs) -> None:
		"""
		Remove batch effects using Harmony

		Args:
			batch_keys: List of names of tensors that indicate batches (e.g. `["Chemistry"]`)
		
		Remarks:
			Uses harmony-pytorch (https://github.com/lilab-bcb/harmony-pytorch)

			By default, this module will overwrite the existing Factors tensor. If you
			want to keep the original factors, use `creates: {Factors: MyFactors}` in the
			recipe to rename the output.
		"""
		super().__init__(**kwargs)
		self.batch_keys = batch_keys

	@requires("Factors", "float32", ("cells", None))
	@creates("Factors", "float32", ("cells", None))
	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		if self.batch_keys is not None and len(self.batch_keys) > 0:
			logging.info(f" Harmony: Harmonizing based on batch keys {self.batch_keys}")
			keys_df = pd.DataFrame.from_dict({k: ws[k][:] for k in self.batch_keys})
			transformed = harmonize(self.Factors[:], keys_df, batch_key=self.batch_keys, tol_harmony=1e-5)
		else:
			logging.info(" Harmony: Skipping because no batch keys were provided")
			transformed = self.Factors[:]
		return transformed
