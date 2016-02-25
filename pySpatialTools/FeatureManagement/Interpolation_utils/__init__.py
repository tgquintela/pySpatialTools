
"""
Interpolation
=============
Module which contains the functions to spatially interpolate features.

TODO
----
- Join both ways into 1.

"""

## Interpolation
from general_interpolation import general_interpolate

## Density assignation
from density_assignation import general_density_assignation
from density_assignation_process import DensityAssign_Process

from density_utils import comparison_densities, clustering_by_comparison,\
    population_assignation_f

from weighting_functions import create_weighted_function
