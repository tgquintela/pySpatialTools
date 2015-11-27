
"""
Region spatial relations
------------------------
Group of functions to compute region-region spatial relations.

"""

import numpy as np
from collections import Counter


def regions_relation_points(locs, regions, retriever, info_ret):
    """Function which computes the spatial relations between regions
    considering the shared neighbourhoods of their points.

    """

    ## 0. Compute needed variables
    regs = np.unique(regions)
    count_reg = Counter(regions)
    n_reg = regs.shape[0]
    map_reg = dict(zip(regs, range(regs.shape[0])))

    ## 1. Count coincidences
    coincidence = np.zeros((n_reg, n_reg))
    for i in xrange(locs.shape[0]):
        neighs = retriever.retrieve_neighs(locs[i, :], info_ret[i], False)[0]
        for j in range(len(neighs)):
            coincidence[map_reg[regions[i]], map_reg[regions[neighs[j]]]] += 1.

    ## 2. Normalization of coincidences (OJO LOS ZEROS)
    for i in xrange(n_reg):
        for j in xrange(n_reg):
            coincidence[i, j] = coincidence[i, j]/(count_reg[i] * count_reg[j])

    return coincidence, regs
