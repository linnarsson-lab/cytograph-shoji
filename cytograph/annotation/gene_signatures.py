import numpy as np

import shoji

from cytograph import Algorithm, creates, requires

import logging
import pickle
import json
from typing import List
from pathlib import Path
from tqdm import trange
import logging


class GeneSignatures(Algorithm):
    def __init__(self, signature_names: List[str], path_to_signatures: str, alpha: float = 0.01, use_cache: bool = False, build_cache: bool = False, allow_invalid_genes: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.all_signature_names = []
        self.signature_names = signature_names
        self.path_to_signatures = Path(path_to_signatures)
        self.alpha = alpha
        self.allow_invalid_genes = allow_invalid_genes
        
        if build_cache:
            use_cache = True
            logging.info("Building signatures cache")
            all_signatures = {}
            for path in self.path_to_signatures.iterdir():
                if path.is_file() and path.suffix == ".json":
                    with path.open() as f:
                        for signame, vals in json.load(f).items():
                            if signame not in self.all_signature_names:
                                self.all_signature_names.append(signame)
                                all_signatures[signame] = vals["geneSymbols"]
            with open(self.path_to_signatures / "signatures.pkl", "wb") as fout:
                pickle.dump(all_signatures, fout)
                
        if use_cache:
            logging.info("Loading gene signatures from cache")
            with open(self.path_to_signatures / "signatures.pkl", "rb") as fin:
                all_signatures = pickle.load(fin)
            self.all_signature_names = list(all_signatures.keys())
            self.signatures = {}
            for signame in self.signature_names:
                self.signatures[signame] = all_signatures[signame]            
        else:
            logging.info("Loading gene signatures from json files")
            self.signatures = {}
            for path in self.path_to_signatures.iterdir():
                if path.is_file() and path.suffix == ".json":
                    with path.open() as f:
                        for signame, vals in json.load(f).items():
                            if signame not in self.all_signature_names:
                                self.all_signature_names.append(signame)
                            if signame in self.signature_names:
                                self.signatures[signame] = vals["geneSymbols"]

    @requires("Expression", "uint16", ("cells", "genes"))
    @requires("Gene", "string", ("genes",))
    @requires("GeneTotalUMIs", "uint32", ("genes",))
    @requires("TotalUMIs", "uint32", ("cells",))
    @creates("SignatureNames", "string", ("signatures",))
    @creates("SignatureScores", "float32", ("cells", "signatures"))
    @creates("SignatureBootstrap", "float32", ("cells", ))
    def fit(self, ws: shoji.WorkspaceManager, save: bool = False) -> np.ndarray:
        if "signatures" in ws:
            del ws.signatures
            del ws.SignatureNames
            del ws.SignatureScores
            del ws.SignatureBootstrap
        ws.signatures = shoji.Dimension(shape=len(self.signatures))
                
        genes = self.Gene[:]
        n_genes = ws.genes.length
        n_cells = ws.cells.length
        total_umi = self.TotalUMIs[:]
        gene_total_umi = ws.GeneTotalUMIs[:]
        gene_mean = gene_total_umi / n_cells
        expression_order = np.argsort(gene_total_umi)
        ordered_genes = genes[expression_order]
        ordered_gene_mean = gene_total_umi[expression_order] / n_cells

        # Report on missing genes
        missing = {}
        for signame in self.signature_names:
            for g in self.signatures[signame]:
                if g not in genes:
                    missing.setdefault(signame, []).append(g)
        for signame, missing_genes in missing.items():
            logging.info(f"{signame} missing genes: {', '.join(missing_genes)}")
            self.signatures[signame] = list(set(self.signatures[signame]) - set(missing_genes))
        if len(missing) > 0:
            if self.allow_invalid_genes:
                logging.warning(f"Missing genes are allowed, but it's better to fix the signature gene list")
            else:
                logging.info(f"Missing genes not allowed (use allow_invalid_genes=True to override, but better to fix the signature gene list)")
                raise ValueError("Missing genes")
            
        n_bins = 30
        bin_size = n_genes // n_bins
        offset = n_genes % n_bins
        logging.info(f"Computing control scores in {n_bins} bins")
        control_scores = np.zeros((n_cells, n_bins))
        for i in trange(n_bins):
            ix = offset + i * bin_size
            selection = np.random.choice(bin_size, size=100, replace=False)
            selected_genes = expression_order[ix:ix + bin_size][selection]
            expression = self.Expression[:, selected_genes]
            expression = (expression.T / total_umi).T * 1000
            expression = expression - np.mean(expression, axis=0)
            control_scores[:, i] = np.sum(expression, axis=1)

        def gene_signature_score(signature, signature_genes):
            selected_genes = np.isin(genes, signature_genes)
            expression = ws.Expression[:, selected_genes]
            expression = (expression.T / total_umi).T * 1000
            expression = expression - np.mean(expression, axis=0)
            score = np.mean(expression, axis=1)
            
            control_score_sum = np.zeros(n_cells)
            for g in signature_genes:
                gene_ix = np.where(ordered_genes == g)[0][0] - offset
                control_score_sum += control_scores[:, gene_ix // bin_size]
            return (score - control_score_sum / len(signature_genes) / 100)
        
        logging.info(f"Computing bootstrap scores for P values")
        bootstrap_scores = []
        for j in trange(100):
            gene_names = np.random.choice(genes, size=100, replace=False)  # Each random signature is without replacement, but replacement will happen across samples
            bootstrap_scores.append(gene_signature_score(signame, gene_names)[:, None])
        
        logging.info(f"Computing signature scores")
        scores = []
        names = []
        for ix in trange(len(self.signatures)):
            signame = self.signature_names[ix]
            gene_names = self.signatures[signame]
            scores.append(gene_signature_score(signame, gene_names)[:, None])
            names.append(signame)
        return np.array(names), np.array(np.hstack(scores)), np.percentile(np.array(bootstrap_scores), 100 * (1 - self.alpha), axis=0).flatten()