import logging
import shoji
import numpy as np
from cytograph import Module


class Aggregate(Module):
	def __init__(self, tensor: str, using: str, into: str, by: str = "Clusters", newdim: str = "clusters", **kwargs) -> None:
		"""
		Aggregate tensors by cluster ID (as given by the Clusters tensor)

		Args:
			tensor: The tensor to aggregate
			using:  The aggregation function ("sum", "count", "first", "nnz", "mean", "variance", or "sd")
			into:   The name of the newly created aggregate tensor
			by:     Optionally, the name of the tensor to group by when aggregating (default: 'Clusters')
			newdim: Optionlly, the name of the newly created (or existing) dimension (default: 'clusters')
		
		Remarks:
			Groups a tensor by the values of another tensor, while applying an aggregation function. For example,
			to calculate the mean expression by cluster:

			Aggregate{tensor: Expression, using: mean, into: MeanExpression}
		"""
		super().__init__(**kwargs)
		self.tensor = tensor
		self.using = using
		self.by = by
		self.into = into
		self.newdim = newdim

	def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
		logging.info(f" Aggregate: Grouping by '{self.by}' using '{self.using}'")
		grouped = ws[:].groupby(self.by)
		tensor = ws._get_tensor(self.tensor)
		if self.using == "sum":
			result = grouped.sum(self.tensor)
		elif self.using == "first":
			result = grouped.first(self.tensor)
		elif self.using == "count":
			result = grouped.count(self.tensor)
		elif self.using == "nnz":
			result = grouped.nnz(self.tensor)
		elif self.using == "mean":
			result = grouped.mean(self.tensor)
		elif self.using == "variance":
			result = grouped.variance(self.tensor)
		elif self.using == "sd":
			result = grouped.sd(self.tensor)
		else:
			raise ValueError(f"Invalid aggregation function '{self.using}'")

		result = result[1][np.argsort(result[0])]
		if save:
			# Create the dimension if needed
			if self.newdim not in ws:
				logging.info(f" Aggregate: Creating dimension '{self.newdim}'")
				ws[self.newdim] = shoji.Dimension(shape=result.shape[0])
			ws[self.into] = shoji.Tensor(result.dtype.name, (self.newdim,) + tensor.dims[1:], inits=result)

		return result
