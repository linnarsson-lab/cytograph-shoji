import numpy as np

import shoji
import numpy as np

from cytograph import Algorithm, creates, requires

import logging


# Noted below the updated gene names (Neftel had some old gene symbols)
MES1 = np.array(["CHI3L1","ANXA2","ANXA1","CD44","VIM","MT2A","C1S","NAMPT","EFEMP1","C1R","SOD2","IFITM3","TIMP1","SPP1","A2M","S100A11","MT1X","S100A10","FN1","LGALS1","S100A16","CLIC1","MGST1","RCAN1","TAGLN2","NPC2","SERPING1","TCIM","EMP1","APOE","CTSB","C3","LGALS3","MT1E","EMP3","SERPINA3","ACTN1","PRDX6","IGFBP7","SERPINE1","PLP2","MGP","CLIC4","GFPT2","GSN","NNMT","TUBA1C","GJA1","TNFRSF1A","WWTR1"])
MES2 = np.array(["HILPDA","ADM","DDIT3","NDRG1","HERPUD1","DNAJB9","TRIB3","ENO2","AKAP12","SQSTM1","MT1X","ATF3","NAMPT","NRN1","SLC2A1","BNIP3","LGALS3","INSIG2","IGFBP3","PPP1R15A","VIM","PLOD2","GBE1","SLC2A3","FTL","WARS1","ERO1A","XPOT","HSPA5","GDF15","ANXA2","EPAS1","LDHA","P4HA1","SERTAD1","PFKP","PGK1","EGLN3","SLC6A6","CA9","BNIP3L","RPL21","TRAM1","UFM1","ASNS","GOLT1B","ANGPTL4","SLC39A14","CDKN1A","HSPA9"])
MES = np.concatenate((MES1, MES2))
# C8orf4 -> TCIM
# WARS -> WARS1
# ERO1L -> ERO1A

AC = np.array(["CST3","S100B","SLC1A3","HEPN1","HOPX","MT3","SPARCL1","MLC1","GFAP","FABP7","BCAN","PON2","METTL7B","SPARC","GATM","RAMP1","PMP2","AQP4","DBI","EDNRB","PTPRZ1","CLU","PMP22","ATP1A2","S100A16","HEY1","PCDHGC3","TTYH1","NDRG2","PRCP","ATP1B2","AGT","PLTP","GPM6B","F3","RAB31","PLPP3","ANXA5","TSPAN7"])
# PPAP2B -> PLPP3

OPC = np.array(["BCAN","PLP1","GPR17","FIBIN","LHFPL3","OLIG1","PSAT1","SCRG1","OMG","APOD","SIRT2","TNR","THY1","PHYHIPL","SOX2-OT","NKAIN4","PLPPR1","PTPRZ1","VCAN","DBI","PMP2","CNP","TNS3","LIMA1","CA10","PCDHGC3","CNTN1","SCD5","P2RX7","CADM2","TTYH1","FGF12","PACC1","NEU4","FXYD6","RNF13","RTKN","GPM6B","LMF1","ALCAM","PGRMC1","PLAAT1","BCAS1","RAB31","PLLP","FABP5","NLGN3","SERINC5","EPB41L2","GPR37L1"])
# LPPR1 -> PLPPR1
# TMEM206 -> PACC1
# HRASLS -> PLAAT1

NPC1 = np.array(["DLL3","DLL1","SOX4","TUBB3","HES6","TAGLN3","NEU4","MARCKSL1","CD24","STMN1","TCF12","BEX1","OLIG1","MAP2","FXYD6","PTPRS","MLLT11","NPPA","BCAN","MEST","ASCL1","BTG2","DCX","NXPH1","JPT1","PFN2","SCG3","MYT1","CHD7","ADGRG1","TUBA1A","PCBP4","ETV1","SHD","TNR","AMOTL2","DBN1","HIP1","ABAT","ELAVL4","LMF1","GRIK2","SERINC5","TSPAN13","ELMO1","GLCCI1","SEZ6L","LRRN1","SEZ6","SOX11"])
NPC2 = np.array(["STMN2","CD24","RND3","NSG2","TUBB3","MIAT","DCX","NSG1","ELAVL4","MLLT11","DLX6-AS1","SOX11","NREP","FNBP1L","TAGLN3","STMN4","DLX5","SOX4","MAP1B","RBFOX2","IGFBPL1","STMN1","JPT1","TMEM161B-AS1","DPYSL3","SEPTIN3","PKIA","ATP1B1","DYNC1I1","CD200","SNAP25","PAK3","NDRG4","KIF5A","UCHL1","ENO2","KIF5C","DDAH2","TUBB2A","LBH","LINC01102","TCF4","GNG3","NFIB","DPYSL5","CRABP1","DBN1","NFIX","CEP170","BLCAP"])
NPC = np.concatenate((NPC1, NPC2))
# HN1 -> JPT1
# GPR56 -> ADGRG1
# HMP19 -> NSG2
# SEPT3 -> SEPTIN3
# LOC150568 -> LINC01102

class NeftelStates(Algorithm):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    @requires("Expression", "uint16", ("cells", "genes"))
    @requires("Gene", "string", ("genes",))
    @requires("GeneTotalUMIs", "uint32", ("genes",))
    @requires("TotalUMIs", "uint32", ("cells",))
    @creates("NeftelScore", "float32", ("cells", 4))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:

        logging.info("Loading genes")
        genes = self.Gene[:]
        n_genes = len(genes)
        logging.info("Loading expression matrix")
        expression = self.Expression[:]
        total_umi = self.TotalUMIs[:]
        logging.info("Normalizing to median total UMI")
        expression = (expression.T / total_umi).T * np.median(total_umi)
        logging.info("Calculating relative expression")
        expression = expression - np.mean(expression, axis=0)
        logging.info("Ordering genes by absolute expression")
        expression_order = np.argsort(ws.GeneTotalUMIs[:])
        genes = genes[expression_order]
        expression = expression[:, expression_order]
        
        logging.info("Computing signature scores")
        bin_size = n_genes // 30

        def gene_signature_score(signature_genes):
            score = np.mean(expression[:, np.isin(genes, signature_genes)], axis=1)
            control_genes = np.zeros(n_genes, dtype=bool)
            for g in signature_genes:
                ix = np.where(genes == g)[0][0]
                bin_ix = ix // bin_size
                random100 = np.random.choice(bin_size, size=100, replace=False) + ix // bin_size * bin_size
                random100 = random100[random100 < n_genes]  # Make sure we don't spill over the end of the gene list
                control_genes[random100] = True
            control_score = np.mean(expression[:, control_genes], axis=1)
            return score - control_score
        
        scores = list(map(gene_signature_score, [AC, MES, NPC, OPC]))
        return np.hstack([x[:, None] for x in scores])