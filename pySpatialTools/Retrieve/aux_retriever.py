
"""
auxiliar retriever
------------------
Auxialiar functions for retrieving.

"""

import numpy as np
from explicit_retrievers import SameEleNeigh
from ..SpatialRelations import DummyRegDistance


def _check_retriever(retriever):
    """
    Conditions (code checker function)
    ----------
    self.retriever has data property
    self.retriever[i].data is the spatial information of the elements.
    self.retriever[i].data has __len__ function.
    self.retriever has _default_ret_val property
    self.__init__ has to call to _initialization()
    """

    ## 0. Functions needed
    def check_requireds(requisits, actual):
        fails = []
        for req in requisits:
            if req not in actual:
                fails.append(req)
        logi = bool(fails)
        return logi, fails

    def creation_mesage(fails, types):
        msg = """The following %s which are required are not in the retriever
        given by the user: %s."""
        msg = msg % (types, str(fails))
        return msg

    ## 1. Constraints
    lista = dir(retriever)
    required_p = ['retriever', '_default_ret_val']
    required_f = ['_retrieve_neighs_spec', '_define_retriever',
                  '_format_output']

    ## 2. Checking constraints
    logi_p, fails_p = check_requireds(required_p, lista)
    logi_f, fails_f = check_requireds(required_f, lista)

    ## 3. Raise Error if it is needed
    if logi_p or logi_f:
        if logi_p and logi_f:
            msg = creation_mesage(fails_p, 'parameters')
            msg = msg + "\n"
            msg += creation_mesage(fails_f, 'functions')
        elif logi_p:
            msg = creation_mesage(fails_p, 'parameters')
        elif logi_f:
            msg = creation_mesage(fails_f, 'functions')
        raise TypeError(msg)


def create_retriever_input_output(regions):
    def remap(neighs_info, regions):
        neighs, dists = neighs_info
        neighs_p, dists_p = [], []
        for i in range(len(neighs)):
            neighs_ip = np.where(regions == neighs[i])[0]
            neighs_p.append(neighs_ip)
            dists_p.append(np.ones(len(neighs_ip)) * dists[i])
        if neighs_p:
            neighs_p, dists_p = np.hstack(neighs_p), np.hstack(dists_p)
        return neighs_p, dists_p

    map_input = lambda self, idxs: np.array([regions[idxs]])
    map_output = lambda self, idxs, neighs_info: remap(neighs_info, regions)
    return map_input, map_output


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
