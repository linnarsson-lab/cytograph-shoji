"""
Algorithms for feature selection and for computing enrichments
"""
from .feature_selection_by_multilevel_enrichment import FeatureSelectionByMultilevelEnrichment
from .feature_selection_by_variance import FeatureSelectionByVariance
from .feature_selection_by_pearson_residuals import FeatureSelectionByPearsonResiduals
from .gsea import GSEA
from .enrichment import Enrichment
from .trinarize import Trinarize
