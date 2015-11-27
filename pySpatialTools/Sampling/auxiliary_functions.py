
"""
Auxialiary functions
--------------------
Auxiliary functions for sampling.

"""


import numpy as np


def weighted_sampling_without_repetition(p_cats, n, weights):
    """Weighted sampling without repetition.

    Parameters
    ----------
    p_cats: numpy.ndarray, shape (n_p)
        the categories of the elements for sample.
    n: int
        number of samples we want.
    weights: numpy.ndarray, shape (n_p)
        the weigths for each element.

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """

    # Needed variables
    n_p = p_cats.shape[0]
    sampling = np.zeros(n_p)

    # Normalization of probs
    if weights is None:
        probs = 1./n_p
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
    weights: numpy.ndarray, shape (n_p)
        the weigths for each element.

    Returns
    -------
    indices: numpy.ndarray, shape(n)
        the indices of the samples.

    """
    # Needed variables
    n_p = p_cats.shape[0]
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
        indices = sampling

    return indices


def weighted_nonfixed_sampling_without_repetition(points_cat, probs, n):
    """Sampling of firtst order statistics. This function performs a non-fixed,
    weighted, disperse sampling from a collection of points. The weights are
    given by the first-order statistics.

    Parameters
    ----------
    points_cat: numpy.ndarray, shape(n_p)
        the category of each point represented as a sequential integers.
    probs: numpy.ndarray, shape(n_cat)
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
