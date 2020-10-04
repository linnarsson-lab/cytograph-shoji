from typing import Tuple, List, Optional
import shoji


class CytographMethod:
	def __init__(self) -> None:
		self._requires: List[Tuple[str, Optional[str], Optional[Tuple[str, ...]]]] = []

	def check(self, ws: shoji.WorkspaceManager, origin: str) -> None:
		for (name, dtype, dims) in self._requires:
			if name not in ws:
				raise AttributeError(f"{origin} requires tensor '{name}'")
			tensor = ws._get_tensor(name)
			if dims is not None and tensor.dims != dims:
					raise ValueError(f"{origin} requires tensor '{name}' with dims='{dims}'")
			elif dtype is not None and tensor.dtype != dtype:
				raise ValueError(f"{origin} requires tensor '{name}' with dtype='{dtype}'")
