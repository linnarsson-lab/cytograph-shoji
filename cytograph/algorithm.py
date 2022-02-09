"""
The Algorithm base class from which all custom algorithms must inherit.
"""
from typing import Tuple, Optional, Callable, Union, Dict
import functools
import numpy as np
import shoji
import fdb
import sys
import logging


class Algorithm:
	def __init__(self, requires: Dict[str, str] = None, creates: Dict[str, str] = None) -> None:
		self.requires = requires if requires is not None else {}
		self.creates = creates if creates is not None else {}
		self.workspace: shoji.WorkspaceManager = None

	def __getattr__(self, name) -> shoji.Tensor:
		return self.__getattribute__(name)


def requires(name: str, dtype: Optional[str], dims: Optional[Tuple[Union[str, int, None], ...]]) -> Callable:
	"""
	Declare the tensor requirements of a Cytograph fit() method

	Usage:
		class Test:
			@requires("Expression", "float32", ("cells", "genes"))
			@requires("Age", "string", ())
			def fit(self, ws) -> None:
				# At this point, the requirements will have been checked
				# and are available as tensors self.Expression and self.Age
	
	Remarks:
		The decorator can only be applied to a method that takes a shoji.WorkspaceManager object as first argument
	"""
	def decorator(func: Callable) -> Callable:
		@functools.wraps(func)
		def wrapper(self, ws: shoji.WorkspaceManager, *args, **kwargs):
			# Check if the output tensor was renamed in the class instance
			tname = name
			if not hasattr(self, "requires") or self.requires is None:
				self.requires = {}
			if name in self.requires:
				tname = self.requires[name]
			else:
				self.requires[name] = name
			setattr(self, name, ws[tname])
			self.workspace = ws
			if tname not in ws:
				raise AttributeError(f"{self.__class__} requires tensor '{tname}'")
			tensor = ws._get_tensor(tname)
			if dims is not None and any(td != d and d is not None for td, d in zip(tensor.dims, dims)):
				raise ValueError(f"{self.__class__} requires tensor '{tname}' with dims='{dims}'")
			elif dtype is not None and tensor.dtype != dtype:
				raise ValueError(f"{self.__class__} requires tensor '{tname}' with dtype='{dtype}'")
			return func(self, ws, *args, **kwargs)
		return wrapper
	return decorator


class ResultHolder:
	def __init__(self, args, stored) -> None:
		self.args = args
		self.stored = stored


def creates(name: str, dtype: str, dims: Tuple[Optional[Union[str, int]], ...], indices: bool = False) -> Callable:
	"""
	Declare the tensors that are created by a Cytograph fit() method, when save=True

	Args:
		name		Name of the created tensor (if it exists, it will be overwritten)
		dtype		Datatype of the tensor
		dims		Dimensions tuple of the tensor
		indices		If true, and if dtype=="bool" and the tensor is rank-1 and values are integers, convert values from indices to a bool vector

	Usage:
		class Test:
			@creates("Expression", "float32", ("cells", "genes"))
			@creates("Age", "string", ())
			def fit(self, ws, save) -> None:
				# body of method
				return (expression, age)  # Note order must match order of decorators
				# If fit() was called with save=True, the return values will be saved as the indicated tensors

	Remarks:
		The decorator can only be applied to a method that takes a shoji.WorkspaceManager object as first argument
	"""
	def decorator(func: Callable) -> Callable:
		@functools.wraps(func)
		def wrapper(self, ws: shoji.WorkspaceManager, *args, **kwargs):
			result = func(self, ws, *args, **kwargs)
			if "save" not in kwargs or not kwargs["save"]:
				return result
			if not isinstance(result, (tuple, list, ResultHolder)):
				result = (result,)
			if not isinstance(result, ResultHolder):
				result = ResultHolder(result, result)
			inits = result.args[-1]
			if indices and dtype == "bool" and np.issubdtype(inits.dtype, np.integer) and len(dims) == 1 and isinstance(dims[0], str):
				# Convert indices to bool vector
				bool_vector = np.zeros(ws[dims[0]].length, dtype="bool")
				bool_vector[inits] = True
				inits = bool_vector
			else:
				try:
					inits = inits.astype("object" if dtype == "string" else dtype)
				except AttributeError:
					raise AttributeError("@creates() decorator handles only numpy ndarrays; use np.array(x, dtype=...) to wrap scalars")
			try:
				# Check if the output tensor was renamed in the class instance
				tname = name
				if hasattr(self, "creates") and self.creates is not None:
					if name in self.creates:
						tname = self.creates[name]
				logging.info(f" [Creating tensor {tname} of shape {inits.shape}]")
				ws[tname] = shoji.Tensor(dtype, dims, inits=np.array(inits))
			except fdb.impl.FDBError as e:
				if e.code == 2101:
					logging.error(f"Result tensor '{name}' was too large to fit in transaction")
					sys.exit(1)
				else:
					raise e
			if len(result.args) == 1:
				if len(result.stored) == 1:
					return result.stored[0]
				return result.stored
			return ResultHolder(result.args[:-1], result.stored)
		return wrapper
	return decorator
