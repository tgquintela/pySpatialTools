
"""
Density utils
-------------
Density functions to complement the study of the spatial density.

"""

import numpy as np


def comparison_densities(density1, density2):
    return np.log(density1/float(density2))


def clustering_by_comparison(density1, density2, Zscore=3.):
    comparison = comparison_densities(density1, density2)
    Zs = np.std(comparison)
    m = np.mean(comparison)
    idxs = np.logical_and(comparison >= m-Zs*Zscore, comparison <= m+Zs*Zscore)
    comparison[idxs] = 0
    return comparison


def population_assignation_f(weights, values):
    """Population function decided. Values has 3dim: population, density and
    area).
    """
    ## Only use population data
    pop_assign = np.dot(values[:, 0], weights)
    return pop_assign