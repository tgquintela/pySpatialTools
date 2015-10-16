
"""
Region spatial relations
------------------------
Group of functions to compute region-region spatial relations.

"""

import numpy as np


def regions_relation_points(locs, regions, retriever, info_ret):
    """Function which computes the spatial relations between regions
    considering the shared neighbourhoods of their points.
    """
    n_reg = np.unique(regions).shape[0]
    coincidence = np.zeros((n_reg, n_reg))
    for i in xrange(locs.shape[0]):
        neighs = retriever.retrieve_neighs(locs[i, :], info_ret[i], False)
        print type(neighs)
        for j in range(len(neighs)):
            coincidence[regions[i], regions[neighs[j]]] += 1
    return coincidence
