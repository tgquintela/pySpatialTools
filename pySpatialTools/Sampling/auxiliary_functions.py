
"""
Auxialiary functions
--------------------
Auxiliary functions for sampling.




TODO
----
All the sampling functions possible

"""


import numpy as np


def weighted_sampling_without_repetition(p_cats, n, weights=None):
    """Weighted sampling without repetition.

    Parameters
    ----------
    p_cats: numpy.ndarray, shape (n_p)
        the categories of the elements for sample.
    n: int
        number of samples we want.
    weights: numpy.ndarray, shape (n_p) (default=None)
        the weigths for each element.

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """

    # Needed variables
    n_p = len(p_cats)
    cats = np.unique(p_cats)
    sampling = np.zeros(n_p)

    # Normalization of probs
    if weights is None:
        probs = 1./n_p*np.ones(len(cats))
    else:
        probs = weights*n/float(weights.sum())

    # Sampling
    n_sampled = 0
    i = 0
    while True:
        i = i % n_p
        if not sampling[i]:
            r = np.random.random()
            if r <= probs[p_cats[i]]:
                sampling[i] = 1
                n_sampled += 1
                if n_sampled == n:
                    break
        i += 1

    # Retrieve indices
    indices = np.where(sampling)[0]

    return indices


def weighted_sampling_with_repetition(p_cats, n, weights=None):
    """Weighted sampling with repetition.

    Parameters
    ----------
    p_cats: numpy.ndarray, shape (n_p)
        the categories of the elements for sample.
    n: int
        number of samples we want.
    weights: numpy.ndarray, shape (n_p) (default=None)
        the weigths for each element.

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """
    # Needed variables
    n_p = len(p_cats)
    sampling = np.zeros(n)

    if weights is None:
        indices = p_cats[np.random.randint(0, n_p, n)]
    else:
        # Normalization of probs
        probs = weights*n/float(weights.sum())
        cum_probs = np.cumsum(probs)

        # Sampling
        i = 0
        for i in xrange(n):
            r = np.random.random()
            for j in range(n_p):
                if r <= cum_probs[j]:
                    sampling[i] = p_cats[j]
                    break

        # Retrieve indices
        indices = sampling.astype(int)

    return indices


def weighted_nonfixed_sampling_without_repetition(points_cat, n, probs=None):
    """Sampling of first order statistics. This function performs a non-fixed,
    weighted, disperse sampling from a collection of points. The weights are
    given by the first-order statistics.

    Parameters
    ----------
    points_cat: numpy.ndarray, shape(n_p)
        the category of each point represented as a sequential integers.
    n: integer
        number of samples we want
    probs: numpy.ndarray, shape(n_cat) (default=None)
        probabilities we want to keep in the sample.

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """

    # Needed variables
    n_p = len(points_cat)
    sampling = np.zeros(n_p)
    cats = np.unique(points_cat)

    # Normalization of probs
    if probs is None:
        probs = np.ones(len(cats))
    probs = probs/float(probs.sum())*(n/float(n_p))*len(probs)
#    probs = probs/(n_p/float(n))*(probs.shape[0]/probs.sum())

    # Sampling
    for i in range(n_p):
        if not sampling[i]:
            r = np.random.random()
            if r <= probs[points_cat[i]]:
                sampling[i] = 1

    # Retrieve indices
    indices = np.where(sampling)[0]

    return indices
