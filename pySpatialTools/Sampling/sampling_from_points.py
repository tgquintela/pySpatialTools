
"""
Sampling_from_points
--------------------
Module which groups functions and classes from sampling from a collection of
points.

"""

import numpy as np
from sklearn.neighbors import KDTree
from auxiliary_functions import weighted_sampling_without_repetition,\
    weighted_sampling_with_repetition
from sampling_from_space import uniform_points_sampling


def weighted_point_sampling(points, n, weights=None):
    """Retrieve a group of points considering a uniform probability of
    retrieving each point from the total collection.

    Parameters
    ----------
    points: numpy.ndarray
        the points in the space selected.
    n: integer
        number of points we want to retrieve in the sample.

    Returns
    -------
    points_s: numpy.ndarray
        the sampled points in the space selected.

    """

    ## 0. Compute needed variables
    n_p = len(points)

    ## 1. Compute indices
    if weights is None:
        indices = np.random.permutation(n_p)[:n]
    else:
        p_cats = np.arange(n_p)
        indices = weighted_sampling_without_repetition(p_cats, n, weights)
    return indices


def uniform_points_points_sampling(limits, points, n):
    """Select the spatial uniform points in the sample by sampling uniform
    spatial points and getting the nearest ones in the available ones.

    Parameters
    ----------
    limits: numpy.ndarray, shape (2, 2)
        the limits of the space. There is the square four limits which defines
        the whole retrievable region.
    points: numpy.ndarray
        the points in the space selected.
    n: int
        the number of samples we want.

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """

    ## 0. Initialize retriever
    retriever = KDTree(points)
    ## 1. Compute spatial uniform points
    points_s = uniform_points_sampling(limits, n)
    ## 2. Get the nearest points in the sample
    result = retriever.query(points_s, k=1)
    indices = result[1]
    indices = indices.astype(int)
    return indices


def weighted_region_points_sampling(discretizor, points, n, weights=None):
    """Select points weighting regions.

    Parameters
    ----------
    discretizor: pySpatialTools.Neighbourhood.discretizor object
        the discretization information.
    weights: numpy.ndarray
        the weights of each region.
    n: int
        the number of points we want to sample.

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """

    ## 0. Compute variables needed
    points_c = discretizor.discretize(points)
    p_cats = np.unique(points_c)
    if weights is not None:
        if len(p_cats) < len(weights):
            weights = weights[p_cats]
        elif len(p_cats) < len(weights):
            pass

    ## 1. Compute regions selected
    regions = weighted_sampling_with_repetition(p_cats, n, weights)

    ## 2. Compute uniform selection from the selected points
    indices = -1*np.ones(n)
    for i in xrange(n):
        logi = points_c == regions[i]
        n_i = logi.sum()
        j = np.random.randint(0, n_i)
        indices[i] = np.where(logi)[0][j]

    indices = indices.astype(int)
    return indices


###############################################################################
############################# Clustering ############################
###############################################################################
def regional_sampling(points, n):
    """Retrieve a group of points which performs a cluster-like region.

    Parameters
    ----------
    points: numpy.ndarray
        the points in the space selected.
    n: integer
        number of points we want to retrieve in the sample.

    Returns
    -------
    points_s: numpy.ndarray
        the sampled points in the space selected.

    """

    ## Select region points
    ## Divide regional points in small squares (magnitude parameter)
    ## Compute how the squares are related and perform a community detection
    pass


def clustering_sampling(points, discretizor, retrievers, info_rets,
                        clustering):
    """Sampling by spatial clustering points.
    """
    points_cl = clustering_regions(points, discretizor, retrievers,
                                   info_rets, clustering)

    return points
