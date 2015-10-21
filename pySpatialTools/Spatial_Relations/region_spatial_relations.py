
"""
Region spatial relations
------------------------
Group of functions to compute region-region spatial relations.

"""

import numpy as np


def regions_relation_points(locs, regions, retriever, info_ret):
    """Function which computes the spatial relations between regions
    considering the shared neighbourhoods of their points.

    TODO
    ----
    - Normalization coincidence
    """
    n_reg = np.unique(regions).shape[0]
    regs = np.unique(regions)
    map_reg = dict(zip(regs, range(regs.shape[0])))
    coincidence = np.zeros((n_reg, n_reg))
    for i in xrange(locs.shape[0]):
        neighs = retriever.retrieve_neighs(locs[i, :], info_ret[i], False)[0]
        for j in range(len(neighs)):
            coincidence[map_reg[regions[i]], map_reg[regions[neighs[j]]]] += 1
    return coincidence
