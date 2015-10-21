
"""
Sampling_from_space
-------------------
Module which sample points from a given space.

"""

import numpy as np


def spatial_uniform_sampling(limits, n):
    """Uniformely retrieve of points from space.
    """
    minis = np.min(limits, axis=0)
    maxis = np.max(limits, axis=0)
    A = np.random.random((n, limits.shape[1]))
    points_s = (maxis - minis)*A + minis

    return points_s


def squared_sampling(limits, n):
    """Uniformely retrieve from a sector.
    """
    pass


def clustering_sampling(points, discretizor, retrievers, info_rets,
                        clustering):
    """Sampling by spatial clustering points.
    """
    points_cl = clustering_regions(points, discretizor, retrievers,
                                   info_rets, clustering)


    return points
