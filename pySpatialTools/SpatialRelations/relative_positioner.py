
"""
Relative_positioner
-------------------
Module which contains the classes method and functions to compute the relative
position of the neighbours.

Requisits
---------
- The output need to be a iterable of elements with __len__ function and the
possibility to index indices range(n_neighs).

"""

from scipy.spatial.distance import cdist


class RelativePositioner:
    """Class method to compute the relative positions of neighbours respect the
    main element.

    TODO: it is only a preliminary version.
    """

    def __init__(self, funct, info_pos=None):
        self.funct = funct
        self.info_pos = info_pos

    def compute(self, loc_i, loc_neighs):
        return self.funct(loc_i, loc_neighs, self.info_pos)


class RelativeRegionPositioner:
    """Class method to compute the relative positions of neighbours respect the
    main element.

    TODO: it is only a preliminary version.
    """

    def __init__(self, funct, info_pos=None):
        self.funct = funct
        self.info_pos = info_pos

    def compute(self, loc_i, loc_neighs, reg_i, reg_neighs):
        return self.funct(loc_i, loc_neighs, reg_i, reg_neighs)


########################### Collection of functions ###########################
###############################################################################
def metric_distances(loc_i, loc_neighs, info_pos={}):
    dist_metric = cdist(loc_i, loc_neighs, **info_pos)
    dist_metric = dist_metric.T
    return dist_metric


def diff_vectors(loc_i, loc_neighs, info_pos={}):
    diff_vects = loc_neighs - loc_i
    return diff_vects
