
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




def f_measure(regs, dists, u_regs):
    ""
    count = Counter(regs)
    counts = np.zeros(u_regs.shape[0])
    counts[count.keys()] = np.array(count.values())
    return counts

def normalization_f(regs, dists, u_regs):
    ""
    return


def prepare_computing_points_distance(locs, discretizor, pars_ret):
    ""
    if type(discretizor) == np.ndarray:
        regions = discretizor.discretize(locs)
    else:
        regions = discretizor
    ret = retriever(locs, **pars_ret)
    return ret, regions


def compute_clustering(regions, retriever, f_measure):
    ""
    n = regions.shape[0]
    u_regs = np.unique(regions)
    measure = np.zeros((u_regs.shape[0], u_regs.shape[0]))
    for i in xrange(n):
        print i
        neighs, dists = retriever.retrieve_neighs(i)
        measure[regions[i], :] += f_measure(regions[neighs], dists, u_regs)
    return measure
