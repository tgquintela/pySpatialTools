
"""
Example game of life
--------------------
Example in which is performed the evolution of the cellular automata following
the Conway's rule.
"""

import numpy as np
from pySpatialTools.base import BaseDescriptorModel
from pySpatialTools.Retrieve import WindowsRetriever, RetrieverManager
from pySpatialTools.FeatureManagement import SpatialDescriptorModel
from pySpatialTools.FeatureManagement.features_objects import\
    ImplicitFeatures
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.utils.mapper_vals_i import create_mapper_vals_i


class ConwayEvolution(BaseDescriptorModel):
    """Conway evolution."""

    def _f_default_names(self, features_o):
        return ['state']

    def compute(self, pointfeats, point_pos):
        descriptors = np.sum(pointfeats, axis=1)
        return descriptors

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        "General default relative descriptors."
        descriptors = desc_i[:]
        ind_life = desc_i.ravel().astype(bool)
        ind_dead = np.logical_not(desc_i.ravel())
        descriptors[ind_life] = np.logical_or(desc_neigh[ind_life] == 2,
                                              desc_neigh[ind_life] == 3)
        descriptors[ind_dead] = desc_neigh[ind_dead] == 3

        return descriptors


if __name__ == "__main__":
    ## Initial variables
    nx, ny, nt = 10, 10, 100
    pars_ret = {'l': 3, 'center': 0, 'excluded': True}
    ## Initialization
    initial_state = np.random.randint(0, 2, nx*ny)
    windret = WindowsRetriever((nx, ny), pars_ret=pars_ret)
    gret = RetrieverManager(windret)
    conw_ev = ConwayEvolution()
    ## Evolution
    state = initial_state[:]
    m_vals_i = create_mapper_vals_i(('matrix', len(state), len(state)))
    for i in range(nt):
        state =\
            FeaturesManager(ImplicitFeatures(state, descriptormodel=conw_ev),
                            maps_vals_i=m_vals_i)
        spdesc = SpatialDescriptorModel(gret, state)
        state = spdesc.compute()[:, :, 0].astype(int)
