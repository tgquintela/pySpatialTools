
"""
Auxiliar retrievers parsing
---------------------------
Tools and utilities to parse heterogenous ways to give retriever information
in order to obtain retriever objects.
"""


import numpy as np
from retrievers import Retriever
from collectionretrievers import RetrieverManager
from pySpatialTools.Discretization import _discretization_parsing_creation


###############################################################################
######################## Main paser creation functions ########################
###############################################################################
################################## Retrievers #################################
def _retrieverobject_parsing_creation(retriever_info):
    """Function which uniforms the retriever info to be useful in other
    parts of the code.

    Standarts
    * Retriever object
    * (Retriever class, main_info)
    * (Retriever class, main_info, pars_ret)
    * (Retriever class, main_info, pars_ret, autolocs)
    """
    if isinstance(retriever_info, Retriever):
        pass
    else:
        assert(type(retriever_info) == tuple)
        assert(isinstance(retriever_info[0], object))
        pars_ret = {}
        if len(retriever_info) >= 3:
            pars_ret = retriever_info[2]
        if len(retriever_info) == 4:
            pars_ret['autolocs'] = retriever_info[3]
        retriever_info = retriever_info[0](retriever_info[1], **pars_ret)
    assert(isinstance(retriever_info, Retriever))
    return retriever_info


def _retrievermanager_parsing_creation(retriever_info):
    """Function which uniforms the retriever info to be useful in other
    parts of the code.

    Standarts
    * Retriever object
    * Retriever objects
    """
    if isinstance(retriever_info, Retriever):
        retriever_info = RetrieverManager(retriever_info)
    elif type(retriever_info) == list:
        assert(all([isinstance(e, Retriever) for e in retriever_info]))
        retriever_info = RetrieverManager(retriever_info)
    else:
        assert(isinstance(retriever_info, RetrieverManager))
    assert(isinstance(retriever_info, RetrieverManager))
    return retriever_info


def _retriever_parsing_creation(retriever_info):
    """Function which uniforms the retriever info to be useful in other
    parts of the code.

    Standarts
    * Retriever object
    * (Retriever class, main_info)
    * (Retriever class, main_info, pars_ret)
    * (Retriever class, main_info, pars_ret, autolocs)
    """
    if isinstance(retriever_info, RetrieverManager):
        pass
    elif isinstance(retriever_info, Retriever):
        retriever_info = _retrievermanager_parsing_creation(retriever_info)
    elif type(retriever_info) == list:
        r = [_retrieverobject_parsing_creation(ret) for ret in retriever_info]
        retriever_info = _retrievermanager_parsing_creation(r)
    else:
        retriever_info = _retrieverobject_parsing_creation(retriever_info)
        retriever_info = _retrievermanager_parsing_creation(retriever_info)
    assert(isinstance(retriever_info, RetrieverManager))
    return retriever_info


###############################################################################
################## Creation of automatic discretization maps ##################
###############################################################################
def create_m_in_inverse_discretization(discretization_info):
    """Create in_map for inverse discretization."""
    ## 0. Parsing discretization information input
    locs, regions, disc = _discretization_parsing_creation(discretization_info)

    ## 1. Building map
    def m_in_inverse_discretazation(self, idxs):
        """Inverse application of the discretization information."""
        new_idxs = []
        for i in idxs:
            new_idxs.append(np.where(regions == i)[0])
        return new_idxs

    return m_in_inverse_discretazation


def create_m_in_direct_discretization(discretization_info):
    """Create in_map for direct discretization."""
    ## 0. Parsing discretization information input
    locs, regions, disc = _discretization_parsing_creation(discretization_info)
    ## 1. Building map
    if disc is not None:
        def m_in_direct_discretazation(self, idxs):
            """Direct application of the discretization information."""
            return [disc.discretize(locs[i]) for i in idxs]
    else:
        def m_in_direct_discretazation(self, idxs):
            """Direct application of the discretization information."""
            return [np.array(regions[e]) for e in idxs]

    return m_in_direct_discretazation


def create_m_out_inverse_discretization(discretization_info):
    """Create out_map for inverse discretization."""
    ## 0. Parsing discretization information input
    locs, regions, disc = _discretization_parsing_creation(discretization_info)

    ## 1. Building map
    if type(regions) == np.ndarray:
        def m_out_inverse_discretization(self, idxs, neighs_info):
            """This out_map for retrievers change the size of neighbourhood by
            substituting the regions_id or groups_id in the neighs_info for the
            elements which belong to this groups.
            """
            neighs, dists = neighs_info
            neighs_o, dists_o = [], []
            for iss_i in range(len(neighs)):
                neighs_p, dists_p = [], []
                for i in range(len(neighs[iss_i])):
                    neighs_ip = np.where(regions == neighs[iss_i][i])[0]
                    neighs_p.append(neighs_ip)
                    if dists[iss_i] is not None:
                        sh = len(neighs_ip), 1
                        dists_p.append(np.ones(sh) * dists[iss_i][i])
                if neighs_p:
                    neighs_p = np.concatenate(neighs_p)
                if dists_p:
                    dists_p = np.concatenate(dists_p)
                else:
                    dists_p = np.ones((0, 1))
                neighs_o.append(neighs_p)
                dists_o.append(dists_p)
            return neighs_o, dists_o

    return m_out_inverse_discretization


def create_m_out_direct_discretization(discretization_info):
    """Create out_map for inverse discretization."""
    ## 0. Parsing discretization information input
    locs, regions, disc = _discretization_parsing_creation(discretization_info)

    ## 1. Building map
    if disc is None:
        def m_out_direct_discretization(self, idxs, neighs_info):
            """This out_map for retrievers don't change the size of
            neighbourhood, only substitutes the element id for the group or
            regions id. It is useful for PhantomFeatures and direct distance
            features. Distance don't change."""
            neighs, dists = neighs_info
            neighs_o = []
            for iss_i in range(len(neighs)):
                neighs_p = []
                for i in range(len(neighs[iss_i])):
                    neighs_ip = np.array([regions[neighs[iss_i][i]]]).ravel()
                    neighs_p.append(neighs_ip)
                if neighs_p:
                    neighs_p = np.concatenate(neighs_p)
                neighs_o.append(neighs_p)
            return neighs_o, dists
    else:
        def m_out_direct_discretization(self, idxs, neighs_info):
            """This out_map for retrievers don't change the size of
            neighbourhood, only substitutes the element id for the group or
            regions id. It is useful for PhantomFeatures and direct distance
            features. Distance don't change."""
            neighs, dists = neighs_info
            neighs_o = []
            for iss_i in range(len(neighs)):
                neighs_p = []
                for i in range(len(neighs[iss_i])):
                    neighs_ip = disc.discretize(locs[neighs[iss_i][i]])
                    neighs_p.append(np.array([neighs_ip]).ravel())
                if neighs_p:
                    neighs_p = np.concatenate(neighs_p)
                neighs_o.append(neighs_p)
            return neighs_o, dists

    return m_out_direct_discretization
