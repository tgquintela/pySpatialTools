

import numpy as np
from retrievers import Retriever


def _discretization_parsing_creation(discretization_info):
    """Function which uniforms the discretization info to be useful in other
    parts of the code.

    Standarts
    * (discretizator, locs)
    * (locs, regions)
    """
    assert(type(discretization_info) == tuple)
    if type(discretization_info[0]).__name__ == 'instance':
        regs = discretization_info[0].discretize(discretization_info[1])
        locs = discretization_info[1]
    else:
        assert(type(discretization_info[1]) == np.ndarray)
        assert(len(discretization_info[0]) == len(discretization_info[1]))
        locs, regs = discretization_info
    return locs, regs


def _retriever_parsing_creation(retriever_info):
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
    return retriever_info
