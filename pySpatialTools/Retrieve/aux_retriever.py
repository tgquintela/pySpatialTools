
"""
auxiliar retriever
------------------
Auxialiar functions for retrieving.

"""

import numpy as np


class DummyRetriever:
    """Dummy retriever container. It gives the structure desired by the
    retrievers classes to work properly.
    """

    def __init__(self, data):
        self.data = data


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
                  '_format_output_exclude', '_format_output_noexclude']

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
