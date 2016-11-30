
"""
Density utils
-------------
Density functions to complement the study of the spatial density.

"""

import numpy as np


def comparison_densities(density1, density2):
    """Log comparison between densities.

    Parameters
    ----------
    density1: float or np.ndarray
        the density assignated to a point.
    density2: float or np.ndarray
        the density assignated to a point.

    Returns
    -------
    comparison: float
        the comparison of densities.

    """
    return np.log(density1/density2)


def clustering_by_comparison(density1, density2, Zscore=3.):
    """Clustering comparison

    Parameters
    ----------
    density1: float
        the density assignated to a point.
    density2: float
        the density assignated to a point.
    Zscore: float
        the filter for the ones closer to a Zscore value.

    Returns
    -------
    comparison: float
        the comparison of densities.

    """
    comparison = comparison_densities(density1, density2)
    Zs = np.std(comparison)
    m = np.mean(comparison)
    idxs = np.logical_and(comparison >= m-Zs*Zscore, comparison <= m+Zs*Zscore)
    comparison[idxs] = 0
    return comparison


def population_assignation_f(weights, values):
    """Population function decided. Values has 3dim: population, density and
    area).

    Parameters
    ----------
    weights: np.ndarray
        the weights of each population quantity.
    values: np.ndarray, shape (n, 3)
        the values of population, density and area.

    Returns
    -------
    pop_assign: float
        the assignation population to the evaluated point.

    """
    ## Only use population data
    pop_assign = np.dot(values[:, 0], weights)
    return pop_assign
