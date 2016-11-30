
"""
Sampling_from_space
-------------------
Module which sample points from a given space.

TODO
----
Clustering Sampling

"""

import numpy as np
from auxiliary_functions import weighted_sampling_with_repetition


def uniform_points_sampling(limits, n):
    """Uniformely retrieve of points from space considering all possible points
    of the region are retrievable.

    Parameters
    ----------
    limits: numpy.ndarray, shape (2, 2)
        the limits of the space. There is the square four limits which defines
        the whole retrievable region.
    n: int
        the number of samples we want.

    Returns
    -------
    points_s: numpy.ndarray, shape (n, 2)
        the coordinates of the sampled points.

    """
    limits = np.array(limits)
    minis = np.min(limits, axis=0)
    maxis = np.max(limits, axis=0)
    A = np.random.random((n, limits.shape[1]))
    points_s = (maxis - minis)*A + minis

    return points_s


def weighted_region_space_sampling(discretizor, n, weights=None):
    """Select points weighting regions.

    Parameters
    ----------
    discretizor: pySpatialTools.Discretization.BaseDiscretizor object
        the discretization information.
    weights: numpy.ndarray
        the weights of each region.
    n: int
        the number of points we want to sample.

    Returns
    -------
    points_s: numpy.ndarray, shape (n, 2)
        the coordinates of the sampled points.

    """

    ## 0. Compute variables needed
    regions = discretizor.regions_id
    #regions.shape[0] == weights.shape[0]

    ## 1. Compute regions selected
    regions = weighted_sampling_with_repetition(regions, n, weights)

    ## 2. Compute uniform selection from the selected points
    t_regions = np.unique(regions)
    points_s = -1*np.ones((n, 2))
    # Region by region
    for r in t_regions:
        # Compute limits of the region
        limits_r = discretizor.get_limits(r)
        # Compute how many points of this region you want
        logi = regions == r
        inds = np.where(logi)[0]
        n_r = logi.sum()
        j = 0
        while j != n_r:
            # Generate points randomly in the limits
            point_j = uniform_points_sampling(limits_r, 1)
            # Reject if there is not in the region, accept otherwise
            boolean = discretizor.belong_region(point_j, r)
            if boolean:
                points_s[inds[j]] = point_j
                j += 1

    return points_s
