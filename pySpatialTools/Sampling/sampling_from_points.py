
"""
Sampling_from_points
--------------------
Module which groups functions and classes from sampling from a collection of
points.

"""

import numpy as np


def global_sampling(points, n):
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
    points_s = np.random.permutation(points)[:n, :]
    return points_s


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

    pass


###############################################################################
############################## WEIGHTED SAMPLING ##############################
###############################################################################
def firstorder_sampling_fixed(points_cat, probs, n):
    """Sampling of firtst order statistics. This function performs a fixed,
    weighted, disperse sampling from a collection of points. The weights are
    given by the first-order statistics.

    Parameters
    ----------
    points_cat: numpy.ndarray, shape(n_p)
        the category of each point represented as a sequential integers.
    probs: numpy.ndarray
        probabilities we want to keep in the sample.
    n: integer
        number of samples we want

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """

    # Needed variables
    n_p = points_cat.shape[0]
    sampling = np.zeros(n_p)

    # Normalization of probs
    probs = probs/(n_p/float(n))

    # Sampling
    n_sampled = 0
    i = 0
    while True:
        i = i % n_p
        if not sampling[i]:
            r = np.random.random()
            if r <= probs[points_cat[i]]:
                sampling[i] = 1
                n_sampled += 1
                if n_sampled == n:
                    break
        i += 1

    # Retrieve indices
    indices = np.where(sampling)[0]

    return indices


def firstorder_sampling_nonfixed(points_cat, probs, n):
    """Sampling of firtst order statistics. This function performs a non-fixed,
    weighted, disperse sampling from a collection of points. The weights are
    given by the first-order statistics.

    Parameters
    ----------
    points_cat: numpy.ndarray, shape(n_p)
        the category of each point represented as a sequential integers.
    probs: numpy.ndarray
        probabilities we want to keep in the sample.
    n: integer
        number of samples we want

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """

    # Needed variables
    n_p = points_cat.shape[0]
    sampling = np.zeros(n_p)

    # Normalization of probs
    probs = probs/(n_p/float(n))

    # Sampling
    for i in range(n_p):
        if not sampling[i]:
            r = np.random.random()
            if r <= probs[points_cat[i]]:
                sampling[i] = 1

    # Retrieve indices
    indices = np.where(sampling)[0]

    return indices


def secondorder_sampling():
    """TODO: not a second order but the whole neighbourhood stats.
    """
    pass
