"""
Algorithms for visualization
"""
from .colors import colors75, colorize, Colorizer, NamedColorScheme, DiscreteColorScheme
from .dendrogram import dendrogram
from .heatmap import Heatmap
from .scatter import scatterc, scattern, scatterm
from .plot_qc import PlotQC
from .plot_batches import PlotBatches
from .plot_manifold import PlotManifold
from .plot_scatter import NumericalScatterplot, CategoricalScatterplot
from .plot_age import PlotAge
from .plot_region import PlotRegion
from .plot_subregion import PlotSubregion
from .plot_cell_cycle import PlotCellCycle
from .plot_markers import PlotMarkers
from .plot_karyotype import PlotKaryotype
from .plot_droplet_classes import PlotDropletClasses
from .plot_overview import PlotOverview, PlotOverviewEEL, PlotOverviewEELGraph
from .plot_annotation import PlotAnnotation
