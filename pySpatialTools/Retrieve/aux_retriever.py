
"""
auxiliar retriever
------------------
Auxialiar functions for retrieving.

"""

import numpy as np


def create_retriever_input_output(regions):
    def remap(neighs_info, regions):
        neighs, dists = neighs_info
        neighs_p, dists_p = [], []
        for i in range(len(neighs)):
            neighs_ip = np.where(regions == neighs[i])[0]
            neighs_p.append(neighs_ip)
            dists_p.append(np.ones(len(neighs_ip)) * dists[i])
        neighs_p, dists_p = np.hstack(neighs_p), np.hstack(dists_p)
        return neighs_p, dists_p

    map_input = lambda idxs: np.array([regions[idxs]])
    map_output = lambda idxs, neighs_info: remap(neighs_info, regions)
    return map_input, map_output
