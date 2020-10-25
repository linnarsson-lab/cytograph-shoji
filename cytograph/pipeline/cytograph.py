import logging
import cytograph as cg
from cytograph import requires
import shoji

from .config import Config


class Cytograph:
	def __init__(self, *, config: Config) -> None:
		"""
		Run cytograph

		Args:
			config			The run configuration
		
		Remarks:
			All parameters are obtained from the config object, which comes from the default config
			and can be overridden by the config in the current punchcard
		"""
		self.config = config

	@requires("Species", "string", ())
	def fit(self, ws: shoji.WorkspaceManager) -> None:
		logging.info(f"Cytograph: Analyzing {ws.cells.length} cells")
		species = cg.Species(ws[:].Species)
		logging.info(f"Cytograph: Species is '{species.name}'")

		logging.info("Cytograph: Recomputing the list of valid genes")
		cg.GeneSummaryStatistics().fit(ws, save=True)
		
		logging.info(f"Cytograph: Feature selection by variance")
		genes = cg.FeatureSelectionByVariance(self.config.params.n_genes, mask=species.mask(ws, self.config.params.mask)).fit(ws, save=True)
		logging.info(f"Cytograph: Selected {genes.shape[0]} genes")

		logging.info(f"Cytograph: Factorization by GLMPCA")
		transformed, _ = cg.GLMPCA(n_factors=self.config.params.n_factors).fit(ws, save=True)

		logging.info(f"Cytograph: Computing RNN manifold")
		cg.Manifold(k=self.config.params.k, metric="euclidean").fit(transformed, save=True)

		logging.info(f"Cytograph: Computing tSNE embedding from GLMPCA latent space")
		cg.ArtOfTsne(metric="euclidean").fit(ws, transformed, save=True)

		logging.info("Cytograph: Clustering by polished Louvain")
		labels = cg.PolishedLouvain().fit(ws, save=True)
		logging.info(f"Cytograph: Found {labels.max() + 1} clusters")
