
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


class BaseRelativePositioner:
    """Class method to compute the relative positions of neighbours respect the
    main element. It visualizes position of element i and we can produce
    heterogeneous and anisotropic metric measures.
    TODO: it is only a preliminary version.
    """

    def __init__(self, funct, info_pos={}):
        """Instantiation of the basic relative positioner.

        Parameters
        ----------
        funct: function
            the function computation and transformation.
        info_pos: optional
            the static parameters of the function.

        """
        self.funct = funct
        self.info_pos = info_pos

    def compute(self, loc_i, loc_neighs):
        """Compute function computes the relative position of all the neighs
        using their spatial information and the spatial information of the
        element neighbourhood retrieved.

        Parameters
        ----------
        loc_i: optional
            the spatial information of the element `i`.
        loc_neighs: list or array_like, len (n)
            the sequencial spatial information of each of the neighbours of
            element `i`.

        Returns
        -------
        sp_neighs: list or array_like, len (n)
            the sequencial spatial relative information of each of the
            neighbours of element `i` which respect to this element `i`.

        """
        return self.funct(loc_i, loc_neighs, self.info_pos)


#class RelativeRegionPositioner:
#    """Class method to compute the relative positions of neighbours respect the
#    main element. It visualizes position of element i and we can produce
#    heterogeneous and anisotropic metric measures.
#
#    TODO: it is only a preliminary version.
#    """
#
#    def __init__(self, funct, info_pos={}):
#        self.funct = funct
#        self.info_pos = info_pos
#
#    def compute(self, loc_i, loc_neighs, reg_i, reg_neighs):
#        return self.funct(loc_i, loc_neighs, reg_i, reg_neighs)


########################### Collection of functions ###########################
###############################################################################
def metric_distances(loc_i, loc_neighs, info_pos={}):
    """Meric distance between the neighs.

    Parameters
    ----------
    loc_i: optional
        the spatial information of the element `i`.
    loc_neighs: list or array_like, len (n)
        the sequencial spatial information of each of the neighbours of
        element `i`.
    info_pos: optional
        the static parameters of the function.

    Returns
    -------
    sp_neighs: list or array_like, len (n)
        the sequencial spatial relative information of each of the
        neighbours of element `i` which respect to this element `i`.

    """
    dist_metric = []
    for i in range(len(loc_i)):
        dist_metric.append(cdist(loc_i[[i]], loc_neighs[i], **info_pos).T)
    return dist_metric


def diff_vectors(loc_i, loc_neighs, info_pos={}):
    """Computation of the substraction of the spatial location of neighbours.

    Parameters
    ----------
    loc_i: optional
        the spatial information of the element `i`.
    loc_neighs: list or array_like, len (n)
        the sequencial spatial information of each of the neighbours of
        element `i`.
    info_pos: optional
        the static parameters of the function.

    Returns
    -------
    sp_neighs: list or array_like, len (n)
        the sequencial spatial relative information of each of the
        neighbours of element `i` which respect to this element `i`.

    """
    diff_vects = []
    for i in range(len(loc_i)):
        diff_vects.append(loc_neighs[i] - loc_i[[i]])
    return diff_vects


###############################################################################
###############################################################################
###############################################################################
###############################################################################
## TO RETRIEVER ??
#def get_individuals(neighs, dists, discretized):
#    "Transform individual regions."
#    logi = np.zeros(len(discretized)).astype(bool)
#    dists_i = np.zeros(len(discretized))
#    for i in range(len(neighs)):
#        logi_i = (discretized == neighs[i]).ravel()
#        logi = np.logical_or(logi, logi_i).ravel()
#        dists_i[logi_i] = dists[i]
#    neighs_i = np.where(logi)[0]
#    dists_i = dists_i[logi]
#    return neighs_i, dists_i
