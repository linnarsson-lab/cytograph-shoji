import logging

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import shoji
from cytograph import Algorithm, requires
from cytograph import scatterm


class PlotGeneSignatures(Algorithm):
    def __init__(self, filename: str = "signatures.png", **kwargs) -> None:
        super().__init__(**kwargs)
        self.filename = filename

    @requires("SignatureNames", "string", ("signatures",))
    @requires("SignatureScores", "float32", ("cells", "signatures"))
    @requires("SignatureBootstrap", "float32", ("cells", ))
    @requires("Embedding", "float32", ("cells", 2))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        scores = self.SignatureScores[:]
        signames = self.SignatureNames[:]
        bs = self.SignatureBootstrap[:]
        xy = self.Embedding[:]
        n_signatures = len(signames)
        n_cols = 4
        n_rows = np.ceil(n_signatures / n_cols)

        # Clip to minimum zero
        scores = np.clip(scores, 0, None)
        plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        vmax = np.percentile(scores, 99)  # Use shared maximum value
        for i, signame in enumerate(signames):
            plt.subplot(n_rows, n_cols, i + 1)
            selected = scores[:, i] > bs
            plt.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=5, vmax=vmax)
            plt.scatter(xy[selected, 0], xy[selected, 1], c=scores[selected, i], s=5, vmin=0, vmax=np.percentile(scores[selected, i], 99), cmap=plt.cm.plasma_r)
            plt.title(signame)
            plt.axis("off")

        if save:
            plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
            plt.close()