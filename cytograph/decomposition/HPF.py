import logging
from typing import Tuple, List, Any

import numpy as np
import scipy.sparse as sparse
from scipy.special import digamma, gammaln, logsumexp
from sklearn.model_selection import train_test_split
from tqdm import trange


def _find_redundant_components(factors: np.ndarray, max_r: float) -> List[int]:
	n_factors = factors.shape[1]
	(row, col) = np.where(np.corrcoef(factors.T) > max_r)
	g = sparse.coo_matrix((np.ones(len(row)), (row, col)), shape=(n_factors, n_factors))
	(n_comps, comps) = sparse.csgraph.connected_components(g)
	non_singleton_comps = np.where(np.bincount(comps) > 1)[0]
	to_randomize: List[int] = []
	for c in non_singleton_comps:
		to_randomize += list(np.where(comps == c)[0][1:])
	return sorted(to_randomize)


def find_redundant_components(beta: np.ndarray, theta: np.ndarray, max_r: float) -> np.ndarray:
	"""
	Figure out which components are redundant (identical to another factor), and
	return them as a sorted ndarray. For each set of redundant factors, all but the
	first element is returned.
	"""
	return np.intersect1d(_find_redundant_components(beta, max_r), _find_redundant_components(theta, max_r))


class HPF:
	"""
	Bayesian Hierarchical Poisson Factorization
	Implementation of https://arxiv.org/pdf/1311.1704.pdf
	"""
	def __init__(
		self,
		k: int,
		*,
		a: float = 0.3,
		b: float = 1,
		c: float = 0.3,
		d: float = 1,
		min_iter: int = 10,
		max_iter: int = 100,
		stop_interval: int = 10,
		epsilon: float = 0.001,
		max_r: float = 0.99,
		compute_X_ppv: bool = True,
		validation_fraction: float = 0) -> None:
		"""
		Args:
			k				Number of components
			a				Hyperparameter a in the paper
			b				Hyperparameter a' in the paper
			c				Hyperparameter c in the paper
			d				Hyperparameter c' in the paper
			max_iter		Maximum number of iterations
			stop_interval	Interval between calculating and reporting the log-likelihood
			epsilon			Fraction improvement required to continue iterating
			max_r			Maximum Pearson's correlation coefficient allowed before a component is considered redundant
			compute_X_ppv	If true, compute the posterior predictive values X_ppv (same shape as X)
		"""
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.min_iter = min_iter
		self.max_iter = max_iter
		self.stop_interval = stop_interval
		self.epsilon = epsilon
		self.max_r = max_r
		self.compute_X_ppv = compute_X_ppv
		self.validation_fraction = validation_fraction

		self.beta: np.ndarray = None
		self.theta: np.ndarray = None
		self.eta: np.ndarray = None
		self.xi: np.ndarray = None
		self.gamma_shape: np.ndarray = None
		self.gamma_rate: np.ndarray = None
		self.lambda_shape: np.ndarray = None
		self.lambda_rate: np.ndarray = None
		self.redundant: np.ndarray = None
		self.validation_data: sparse.coo_matrix = None

		self.X_ppv: np.ndarray = None
		self.log_likelihoods: List[float] = []

		self._tau_rate: np.ndarray = None
		self._tau_shape: np.ndarray = None
		self._lambda_rate: np.ndarray = None
		self._lambda_shape: np.ndarray = None

	def fit(self, X: sparse.coo_matrix) -> Any:
		"""
		Fit an HPF model to the data matrix

		Args:
			X	Data matrix, shape (n_cells, n_genes)

		Remarks:
			After fitting, the factor matrices beta and theta are available as self.theta of shape
			(n_cells, k) and self.beta of shape (k, n_genes)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		(beta, theta, eta, xi, gamma_shape, gamma_rate, lambda_shape, lambda_rate) = self._fit(X)

		self.beta = beta
		self.theta = theta
		self.eta = eta
		self.xi = xi
		self.gamma_rate = gamma_rate
		self.gamma_shape = gamma_shape
		self.lambda_rate = lambda_rate
		self.lambda_shape = lambda_shape
		# Identify redundant components
		self.redundant = find_redundant_components(beta, theta, self.max_r)
		# Posterior predictive distribution
		if self.compute_X_ppv:
			self.X_ppv = theta @ beta.T
		return self

	def _fit(self, X: sparse.coo_matrix, beta_precomputed: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		# Create local variables for convenience
		(n_users, n_items) = X.shape
		k = self.k
		(u, i, y) = (X.row, X.col, X.data)  # u and i are indices of the nonzero entries; y are the values of those entries
		(a, b, c, d) = (self.a, self.b, self.c, self.d)
		logging.info(f"nnz={len(u)}")

		# Compute hyperparameters bp and dp
		def mean_var_prior(X: np.ndarray, axis: int) -> float:
			temp = X.sum(axis=axis)
			return np.mean(temp) / np.var(temp)
		bp = b * mean_var_prior(X, axis=1)
		dp = d * mean_var_prior(X, axis=0)

		# Create the validation dataset
		if self.validation_fraction > 0:
			(u, vu, i, vi, y, vy) = train_test_split(u, i, y, train_size=1 - self.validation_fraction)
			self.validation_data = sparse.coo_matrix((vy, (vu, vi)), shape=X.shape)
		else:
			(vu, vi, vy) = (u, i, y)

		# Initialize the variational parameters with priors and some randomness
		kappa_shape = np.full(n_users, b + k * a, dtype='float32')  # This is actually the first variational update, but needed only once
		kappa_rate = np.random.uniform(0.5 * bp, 1.5 * bp, n_users).astype('float32')
		gamma_shape = np.random.uniform(0.5 * a, 1.5 * a, (n_users, k)).astype('float32')
		gamma_rate = np.random.uniform(0.5 * b, 1.5 * b, (n_users, k)).astype('float32')

		if beta_precomputed:
			tau_shape = self._tau_shape
			tau_rate = self._tau_rate
			lambda_shape = self._lambda_shape
			lambda_rate = self._lambda_rate
		else:
			tau_shape = np.full(n_items, d + k * c, dtype='float32')  # This is actually the first variational update, but needed only once
			tau_rate = np.random.uniform(0.5 * dp, 1.5 * dp, n_items).astype('float32')
			lambda_shape = np.random.uniform(0.5 * c, 1.5 * c, (n_items, k)).astype('float32')
			lambda_rate = np.random.uniform(0.5 * d, 1.5 * d, (n_items, k)).astype('float32')

		self.log_likelihoods = []
		with trange(self.max_iter + 1) as t:
			t.set_description(f"HPF.fit(X.shape={X.shape})")
			for n_iter in t:
				# Compute y * phi only for the nonzero values, which are indexed by u and i in the sparse matrix
				# phi is calculated on log scale from expectations of the gammas, hence the digamma and log terms
				# Shape of phi will be (nnz, k)
				# TODO: rewrite in numba so as to calculate each nnz (and sum over k) without materializing the whole phi matrix
				# TODO: maybe parallelize too?
				phi = (digamma(gamma_shape) - np.log(gamma_rate))[u, :] + (digamma(lambda_shape) - np.log(lambda_rate))[i, :]
				# Multiply y by phi normalized (in log space) along the k axis
				y_phi = y[:, None] * np.exp(phi - logsumexp(phi, axis=1)[:, None])

				# Upate the variational parameters corresponding to theta (the users)
				# Sum of y_phi over users, for each k
				y_phi_sum_u = np.zeros((n_users, k))
				for ix in range(k):
					y_phi_sum_u[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=1).A.T[0]
				gamma_shape = a + y_phi_sum_u
				gamma_rate = (kappa_shape / kappa_rate)[:, None] + (lambda_shape / lambda_rate).sum(axis=0)
				kappa_rate = (b / bp) + (gamma_shape / gamma_rate).sum(axis=1)

				if not beta_precomputed:
					# Upate the variational parameters corresponding to beta (the items)
					# Sum of y_phi over items, for each k
					y_phi_sum_i = np.zeros((n_items, k))
					for ix in range(k):
						y_phi_sum_i[:, ix] = sparse.coo_matrix((y_phi[:, ix], (u, i)), X.shape).sum(axis=0).A
					lambda_shape = c + y_phi_sum_i
					lambda_rate = (tau_shape / tau_rate)[:, None] + (gamma_shape / gamma_rate).sum(axis=0)
					tau_rate = (d / dp) + (lambda_shape / lambda_rate).sum(axis=1)

				if n_iter % self.stop_interval == 0:
					# Compute the log likelihood and assess convergence
					# Expectations
					egamma = gamma_shape / gamma_rate
					elambda = lambda_shape / lambda_rate
					# Sum over k for the expectations
					# This is really a dot product but we're only computing it for the nonzeros (indexed by u and i)
					s = (egamma[vu] * elambda[vi]).sum(axis=1)
					# We use gammaln to compute the log factorial, hence the "y + 1"
					log_likelihood = np.sum(vy * np.log(s) - s - gammaln(vy + 1))
					self.log_likelihoods.append(log_likelihood)

					# Check for convergence
					# TODO: allow for small fluctuations
					if len(self.log_likelihoods) > 1:
						prev_ll = self.log_likelihoods[-2]
						diff = (log_likelihood - prev_ll) / abs(prev_ll)
						t.set_postfix(ll=log_likelihood, diff=diff)
						if diff < self.epsilon and n_iter >= self.min_iter:
							break
					else:
						t.set_postfix(ll=log_likelihood)

		# End of the main fitting loop
		if not beta_precomputed:
			# Save these for future use in self.transform()
			self._tau_shape = tau_shape
			self._tau_rate = tau_rate
			self._lambda_shape = lambda_shape
			self._lambda_rate = lambda_rate

		# Compute beta and theta, which are given by the expectations, i.e. shape / rate
		beta = lambda_shape / lambda_rate
		theta = gamma_shape / gamma_rate
		eta = tau_shape / tau_rate
		xi = kappa_shape / kappa_rate
		return (beta, theta, eta, xi, gamma_shape, gamma_rate, lambda_shape, lambda_rate)

	def transform(self, X: sparse.coo_matrix) -> np.ndarray:
		"""
		Transform the data matrix using an already fitted HPF model

		Args:
			X      Data matrix, shape (n_cells, n_genes)

		Returns:
			Factor matrix theta of shape (n_cells, k)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		(_, theta, _, _, _, _, _, _) = self._fit(X, beta_precomputed=True)

		return theta
