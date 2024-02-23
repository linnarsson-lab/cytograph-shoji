import logging

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

import shoji
from cytograph import Algorithm, requires
from .scatter import scatterm


class PlotNeftel(Algorithm):
    def __init__(self, filename: str = "neftel.png", **kwargs) -> None:
        super().__init__(**kwargs)
        self.filename = filename

    @requires("SignatureNames", "string", ("signatures",))
    @requires("SignatureScores", "float32", ("cells", "signatures"))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        signature_names = self.SignatureNames[:]
        if np.isin(signature_names, ["NEFTEL_AC", "NEFTEL_AC", "NEFTEL_AC", "NEFTEL_AC"]).sum() != 4:
            raise ValueError("One or more NEFTEL_xx signature is missing (run the GeneSignatures algorithm first to fix)")
        ac = self.SignatureScores[:, signature_names == "NEFTEL_AC"]
        mes = self.SignatureScores[:, signature_names == "NEFTEL_MES"]
        npc = self.SignatureScores[:, signature_names == "NEFTEL_NPC"]
        opc = self.SignatureScores[:, signature_names == "NEFTEL_OPC"]

        #Calc x, y coordinates for plotting the meta-modules:
        D = np.maximum(opc, npc) - np.maximum(ac, mes)
        Y = np.log2(abs(D)+1)
        Y[D<0] = -Y[D<0]

        #For the X-axis, if D > 0, calculate OPC vs NPC
        X1 = np.log2(abs((opc - npc)[D>0])+1)
        inv = np.where((npc > opc)[D>0])[0]
        X1[inv] = -X1[inv]

        #For the X-axis, if D < 0, calculate AC vs MES
        X2 = np.log2(abs((ac - mes)[D<0])+1)
        inv = np.where((mes > ac)[D<0])[0]
        X2[inv] = -X2[inv]
        X_sum = np.concatenate((X1,X2))
        Y_sum = np.concatenate((Y[D>0],Y[D<0]))
        
        plt.figure(figsize=(18, 9))
        plt.subplot(121)
        plt.scatter(-X_sum, Y_sum, s=10, lw=0, color='black')
        plt.axvline(0, ls='--', lw=.4)
        plt.axhline(0, ls='--', lw=.4)
        plt.gca().set_aspect('equal')
        max_val = np.max([np.max(X_sum), np.max(-X_sum), np.max(Y_sum), np.max(-Y_sum)])
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
        i = max_val / 2
        y = [i,-i,i,-i]
        z = [-i,-i,i,i]
        n = ["OPC-like", "AC-like","NPC-like","MES-like"]

        for i, txt in enumerate(n):
            plt.annotate(txt, (z[i], y[i]),fontsize=15, ha="center")

        plt.subplot(122)
        scatterm(ws.Embedding[:], c=[ac, mes, npc, opc], cmaps=["gold", "brown", "deepskyblue", "green"], labels=["AC", "MES", "NPC", "OPC"])

        if save:
            plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
            plt.close()