
"""
Retriever tools
---------------
Tools to use retrievers

"""

import numpy as np
from explicit_retrievers import SameEleNeigh
from ..SpatialRelations import DummyRegDistance


def create_aggretriever(discretization, regmetric=None, retriever=None,
                        pars_retriever={}):
    """Only it is useful this function if there is only one retriever
    previously and we are aggregating the first one.
    """
    ## 0. Preparations of discretization
    if type(discretization) == np.ndarray:
        def map_in(i):
            if type(i) != int:
                raise TypeError("It is required integer index.")
            return discretization[i]
        u_regs = np.unique(discretization)
    else:
        def map_in(loc):
            return discretization.discretize(loc)
        u_regs = np.unique(discretization.regions_id)
    ## 1. Region Metrics
    if regmetric is None:
        regmetric = DummyRegDistance(u_regs)
    if retriever is None:
        retriever = SameEleNeigh(regmetric, **pars_retriever)
    else:
        retriever = retriever(regmetric, **pars_retriever)

    retriever._input_map = map_in

    return retriever
