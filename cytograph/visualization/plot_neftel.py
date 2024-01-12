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

    @requires("NeftelScore", "float32", ("cells", 4))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> None:
        ac, mes, npc, opc = np.hsplit(self.NeftelScore[:], 4)

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
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        y = [5.5,-5.5,5.5,-5.5]
        z = [-5.5,-5.5,5.5,5.5]
        n = ["OPC-like", "AC-like","NPC-like","MES-like"]

        for i, txt in enumerate(n):
            plt.annotate(txt, (z[i], y[i]),fontsize=15, ha="center")

        plt.subplot(122)
        scatterm(ws.Embedding[:], c=[ac, mes, npc, opc], cmaps=["gold", "brown", "deepskyblue", "green"], labels=["AC", "MES", "NPC", "OPC"])

        if save:
            plt.savefig(self.export_dir / (ws._name + "_" + self.filename), dpi=300, bbox_inches='tight')
            plt.close()