import torch
from tqdm import trange
import numpy as np


class Umify:
	def __init__(self, x: np.ndarray, device: str = "cpu") -> None:
		n_cells, n_genes = x.shape
		self.u = torch.tensor(np.random.uniform(-1, 1, size=(n_cells,1)), device=device, requires_grad=True, dtype=torch.float32)
		self.v = torch.tensor(np.random.uniform(-1, 1, size=(1, n_genes)), device=device, requires_grad=True, dtype=torch.float32)
		self.expected_lambda = -torch.log(torch.tensor(1 - np.count_nonzero(x, axis=0) / n_cells, device=device, dtype=torch.float32))
		self.x = torch.tensor(x.astype("float32"), device=device, dtype=torch.float32)
		self.losses = []

	def _loss_fn(self) -> np.ndarray:
		n_cells, n_genes = self.x.shape
		lambda_ = (self.x * torch.nn.Sigmoid()(self.u).expand(-1, n_genes) * torch.nn.Sigmoid()(self.v).expand(n_cells, -1)).mean(axis=0)
		loss = torch.nn.MSELoss()(lambda_, self.expected_lambda)
		return loss

	def fit(self, num_epochs: int = 100, learning_rate: float = 0.1) -> "BoneFight":
		"""
		Fit the Umify model

		Args:
			num_epochs		Number of epochs
			learning_rate	Learning rate (default: 0.1)
		"""
		optimizer = torch.optim.Adam([self.u, self.v], lr=learning_rate)
		with trange(num_epochs) as t:
			for _ in t:
				loss = self._loss_fn()
				lossn = loss.detach().numpy()
				t.set_postfix(loss=lossn)
				self.losses.append(lossn)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		u = torch.nn.Sigmoid()(self.u.detach()).numpy()
		v = torch.nn.Sigmoid()(self.v.detach()).numpy()
	
		# Rescale so that u.mean() is 1
		return u / u.mean(), v * u.mean()