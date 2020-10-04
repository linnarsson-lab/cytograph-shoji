import numpy as np
import scipy.sparse as sparse


# TODO: (see https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/similarity_based/lap_score.py)
def laplacian_score(X: np.ndarray, W: sparse.coo_matrix) -> np.ndarray:
	"""
	This function implements the laplacian score feature selection, steps are as follows:
	1. Construct the affinity matrix W if it is not specified
	2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
	3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
	4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat)
	Input
	-----
	X: {numpy array}, shape (n_samples, n_features)
		input data
	W: {sparse matrix}, shape (n_samples, n_samples)
		input affinity matrix

	Output
	------
	score: {numpy array}, shape (n_features,)
		laplacian score for each feature

	Reference
	---------
	He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
	"""

	# build the diagonal D matrix from affinity matrix W
	D = np.array(W.sum(axis=1))
	L = W
	tmp = np.dot(np.transpose(D), X)
	D = sparse.diags(np.transpose(D), [0])
	Xt = np.transpose(X)
	t1 = np.transpose(np.dot(Xt, D.todense()))
	t2 = np.transpose(np.dot(Xt, L.todense()))
	# compute the numerator of Lr
	D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp) / D.sum()
	# compute the denominator of Lr
	L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp) / D.sum()
	# avoid the denominator of Lr to be 0
	D_prime[D_prime < 1e-12] = 10000

	# compute laplacian score for all features
	score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]
	return np.transpose(score)
