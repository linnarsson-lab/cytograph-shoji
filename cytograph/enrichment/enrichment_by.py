import numpy as np
from numpy_groupies.aggregate_numpy import aggregate
from sklearn.preprocessing import LabelEncoder
from tqdm import trange
import logging
from cytograph import requires, creates, Algorithm
import shoji


class EnrichmentBy(Algorithm):
    def __init__(self, group: str, **kwargs) -> None:
        """
        Compute gene enrichment in groups defined by a categorical tensor

        Remarks:
            Gene enrichment is computed as the regularized fold-change between
            each group and all other groups. 
        """
        super().__init__(**kwargs)
        self.group = group

    @requires("Expression", "uint16", ("cells", "genes"))
    @creates("EnrichmentByGroup", "float32", (None, "genes"))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
        logging.info(f" EnrichmentBy: Computing enrichment by {self.group}")
        n_cells, n_genes = self.Expression.shape
        
        labels = ws[self.group][:]
        le = LabelEncoder()
        le.fit(labels)
        groups = le.classes_
        n_groups = len(groups)
        
        group_size = np.array([(labels == group).sum() for group in groups])
        means = np.zeros((n_genes, n_groups))
        nnz = np.zeros((n_genes, n_groups))
        enrichment = np.zeros((n_genes, n_groups))
        
        logging.info(f" EnrichmentBy: Calculating means and number of nonzeros per group")
        batch_size = 1000
        for ix in trange(0, n_genes, batch_size):
            x = self.Expression[:, ix: ix + batch_size]
            means[ix: ix + batch_size, :] = aggregate(le.transform(labels), x, func='mean', size=n_groups, axis=0).T
            nnz[ix: ix + batch_size, :] = aggregate(le.transform(labels), x, func=np.count_nonzero, size=n_groups, axis=0).T
        
        logging.info(f" EnrichmentBy: Calculating enrichment scores")
        f_nnz = nnz / group_size
        enrichment = np.zeros_like(means)
        for j in trange(n_groups):
            ix = np.arange(n_groups) != j
            weights = group_size[ix] / group_size[ix].sum()
            means_other = np.average(means[:, ix], weights=weights, axis=1)
            f_nnz_other = np.average(f_nnz[:, ix], weights=weights, axis=1)
            enrichment[:, j] = (f_nnz[:, j] + 0.1) / (f_nnz_other + 0.1) * (means[:, j] + 0.01) / (means_other + 0.01)
        enrichment = enrichment.T
        return enrichment

