
"""
Retriever tools
---------------
Tools to use retrievers

"""

import numpy as np
from pySpatialTools.Discretization import _discretization_parsing_creation
from retrievers import BaseRetriever


###############################################################################
############################# Create aggretriever #############################
###############################################################################
def create_aggretriever(aggregation_info):
    """This function works to aggregate a retriever following the instructions
    input in the aggregation_info variable. It returns an instance of a
    retriever object to be appended in the collection manager list of
    retrievers.

    Parameters
    ----------
    aggregation_info: tuple
        the information to create a retriever aggregation.

    Returns
    -------
    ret_out: pst.BaseRetriever
        the retriever instance.

    """
    ## 0. Preparing inputs
    assert(type(aggregation_info) == tuple)
    disc_info, _, retriever_out, agg_info = aggregation_info
    assert(type(agg_info) == tuple)
    assert(len(agg_info) == 2)
    aggregating_ret, _ = agg_info
    ## 1. Computing retriever_out
    locs, regs, disc = _discretization_parsing_creation(disc_info)
    ret_out = aggregating_ret(retriever_out, locs, regs, disc)
    assert(isinstance(ret_out, BaseRetriever))
    return ret_out


###############################################################################
################### Candidates to aggregating_out functions ###################
###############################################################################
def dummy_implicit_outretriver(retriever_out, locs, regs, disc):
    """Dummy implicit outretriever creation. It only maps the common output
    to a regs discretized space.

    Parameters
    ----------
    retriever_out: class (pst.BaseRetriever)
        the retriever object.
    locs: list, np.ndarray or other
        the spatial information of the retrievable elements.
    regs: np.ndarray
        the assigned region for each of the retrievable spatial elements.
    disc: pst.BaseDiscretizor
        a discretizor.

    Returns
    -------
    ret_out: pst.BaseRetriever
        the retriever instance.

    """

    ## Assert inputs
    assert(type(retriever_out) == tuple)
    assert(isinstance(retriever_out[0], object))

    ## Preparing function output and pars_ret
    def m_out(self, i_locs, neighs_info):
        neighs, dists = neighs_info
        for i in range(len(neighs)):
            for nei in range(len(neighs[i])):
                neighs[i][nei] = regs[neighs[i][nei]]
        return neighs, dists
    pars_ret = {}
    if len(retriever_out) == 2:
        pars_ret = retriever_out[1]
    pars_ret['output_map'] = m_out

    ## Instantiation
    ret_out = retriever_out[0](locs, **pars_ret)
    assert(isinstance(ret_out, BaseRetriever))
    return ret_out


def dummy_explicit_outretriver(retriever_out, locs, regs, disc):
    """Dummy explicit outretriever creation. It computes a regiondistances
    between each regions.

    Parameters
    ----------
    retriever_out: tuple (class (pst.BaseRetriever), dict, function)
        the retriever information.
    locs: list, np.ndarray or other
        the spatial information of the retrievable elements.
    regs: np.ndarray
        the assigned region for each of the retrievable spatial elements.
    disc: pst.BaseDiscretizor
        a discretizor.

    Returns
    -------
    ret_out: pst.BaseRetriever
        the retriever instance.

    """
    ## Assert inputs
    assert(type(retriever_out) == tuple)
    assert(isinstance(retriever_out[0], object))

    pars_ret = {}
    if len(retriever_out) == 2:
        pars_ret = retriever_out[1]
    main_mapper = retriever_out[2](retriever_out[3])

    ret_out = retriever_out[0](main_mapper, **pars_ret)
    assert(isinstance(ret_out, BaseRetriever))
    return ret_out


def avgregionlocs_outretriever(retriever_out, locs, regs, disc):
    """Retriever creation for avg region locations. It retrieves the
    prototype of the region, the average location of the region each one
    belong.

    Parameters
    ----------
    retriever_out: class (pst.BaseRetriever)
        the retriever object.
    locs: list, np.ndarray or other
        the spatial information of the retrievable elements.
    regs: np.ndarray
        the assigned region for each of the retrievable spatial elements.
    disc: pst.BaseDiscretizor
        a discretizor.

    Returns
    -------
    ret_out: pst.BaseRetriever
        the retriever instance.

    """
    u_regs = np.unique(regs)
    avg_locs = np.zeros((len(u_regs), locs.shape[1]))
    for i in xrange(len(u_regs)):
        avg_locs[i] = np.mean(locs[regs == regs[i]], axis=0)
    ret_out = retriever_out[0](avg_locs, **retriever_out[1])
    return ret_out
