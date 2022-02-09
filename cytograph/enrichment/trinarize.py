from math import exp, lgamma, log
from scipy.special import betainc, betaln
from cytograph import requires, creates, Algorithm
import numpy as np
import shoji


class Trinarize(Algorithm):
	"""
	Compute trinarization probability per cluster
	"""
	def __init__(self, f: float = 0.2, **kwargs) -> None:
		"""
		Compute trinarization probability per cluster

		Args:
			f:    The trinarization expression threshold
		
		Remarks:
			Computes the probability that at least a fraction f of the
			cells in a cluster express the gene, given the size of the cluster
			and the observed number of non-zero values.
		"""
		super().__init__(**kwargs)
		self.f = f
		self.p_half_vectorized = np.vectorize(self.p_half)

	@requires("Nonzeros", "uint64", ("clusters", "genes"))
	@requires("NCells", "uint64", ("clusters",))
	@creates("Trinaries", "float32", ("clusters", "genes"))
	def fit(self, ws: shoji.Workspace, save: bool = False) -> np.ndarray:
		k = self.Nonzeros[:]
		n = self.NCells[:]

		p = self.p_half_vectorized(k.T.astype("float32"), n.astype("float32"), self.f).T.astype("float32")
		return p

	def p_half(self, k: int, n: int, f: float) -> float:
		"""
		Return probability that at least half the cells express, if we have observed k of n cells expressing

		Args:
			k (int):	Number of observed positive cells
			n (int):	Total number of cells

		Remarks:
			Probability that at least a fraction f of the cells express, when we observe k positives among n cells is:

				p|k,n = 1-(betainc(1+k, 1-k+n, f)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n))/beta(1+k, 1-k+n)

		Note:
			The formula was derived in Mathematica by computing

				Probability[x > f, {x \[Distributed] BetaDistribution[1 + k, 1 + n - k]}]
		"""

		# These are the prior hyperparameters beta(a,b)
		a = 1.5
		b = 2

		# We really want to calculate this:
		# p = 1-(betainc(a+k, b-k+n, 0.5)*beta(a+k, b-k+n)*gamma(a+b+n)/(gamma(a+k)*gamma(b-k+n)))
		#
		# But it's numerically unstable, so we need to work on log scale (and special-case the incomplete beta)

		incb = betainc(a + k, b - k + n, f)
		if incb == 0:
			p = 1.0
		else:
			try:
				p = 1.0 - exp(log(incb) + betaln(a + k, b - k + n) + lgamma(a + b + n) - lgamma(a + k) - lgamma(b - k + n))
			except ValueError:
				print(k, n)
		return p
