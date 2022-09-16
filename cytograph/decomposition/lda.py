import numpy as np
import tomotopy as tt
from ..utils import available_cpu_count
import scipy.sparse as sparse

# Wrapper around the tomotopy LDAModel, with sane API
class LDA:
	def __init__(self, k: int, genes: np.ndarray, alpha: float, eta: float) -> None:
		self.k = k
		self.genes = genes
		self.alpha = alpha
		self.eta = eta
		self.model: tt.LDAModel = None
		self.X_fitted = None

	def _make_corpus(self, X:np.ndarray) -> tt.utils.Corpus:
		corpus = tt.utils.Corpus()
		if sparse.issparse(X):
			assert isinstance(X, sparse.csr_matrix), "Sparse input matrix must be sparse.csr_matrix"
			for i in range(X.shape[0]):
				corpus.add_doc(list(np.repeat(self.genes, X[i, :].A[0])))
		else:
			for i in range(X.shape[0]):
				corpus.add_doc(list(np.repeat(self.genes, X[i, :])))
		return corpus
	
	def fit(self, X: np.ndarray) -> "LDA":
		self.X_fitted = X
		corpus = self._make_corpus(X)
		self.model = tt.LDAModel(k=self.k, alpha=self.alpha, eta=self.eta)
		self.model.add_corpus(corpus)
		last_perp = 0
		for i in np.arange(0, 500, 10):
			self.model.train(iter=10, workers=available_cpu_count(), parallel=tt.ParallelScheme.PARTITION)
			perp = self.model.perplexity
			if np.abs((last_perp - perp) / perp) < 0.01:
				break
			last_perp = perp

		# Save the unnormalized topics, ordered like the gene list
		self.topics = np.vstack([self.model.get_topic_word_dist(i, normalize=False) for i in range(self.k)]).T
		vocab = np.array(list(self.model.vocabs))
		a = np.array(self.model.vocabs)
		b = self.genes
		indices = a.argsort()[b.argsort().argsort()]
		self.topics = self.topics[indices, :]

		return self

	def transform(self, X: np.ndarray = None) -> np.ndarray:
		if X is None or X is self.X_fitted:
			return np.vstack([doc.get_topic_dist() for doc in self.model.docs])
		else:
			corpus = self._make_corpus(X)
			result = self.model.infer(corpus, workers=available_cpu_count(), parallel=tt.ParallelScheme.PARTITION)
			return np.vstack([doc.get_topic_dist() for doc in result.docs])

	def fit_transform(self, X: np.ndarray) -> np.ndarray:
		self.fit(X)
		return self.transform()

