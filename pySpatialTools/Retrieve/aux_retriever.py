
"""
auxiliar retriever
------------------
Auxialiar functions for retrieving.

"""

import numpy as np


class NullRetriever:
    """Dummy null retriever container. It gives the structure desired by the
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
    required_f = ['_define_retriever', '_format_output_exclude',
                  '_format_output_noexclude']

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

    map_input = lambda self, idxs: [np.array([regions[e]]) for e in idxs]
    map_output = lambda self, idxs, neighs_info: remap(neighs_info, regions)
    return map_input, map_output


############################## Exclude functions ##############################
###############################################################################
def _list_autoexclude(to_exclude_elements, neighs, dists):
    """Over-writable function to exclude the own elements from the retrieved
    set.
    * neighs {list form} [iss][nei]
    * dists {listform} [iss][nei][dim]
    """
    neighs, dists = list(neighs), list(dists) if dists is not None else dists
    for iss_i in range(len(to_exclude_elements)):
        n_iss_i = len(neighs[iss_i])
        idxs_exclude = [i for i in xrange(n_iss_i)
                        if neighs[iss_i][i] in to_exclude_elements[iss_i]]
        neighs[iss_i] = [neighs[iss_i][i] for i in xrange(n_iss_i)
                         if i not in idxs_exclude]
        if dists is not None:
            if n_iss_i:
                aux = [dists[iss_i][i] for i in xrange(n_iss_i)
                       if i not in idxs_exclude]
                dists[iss_i] = aux
            else:
                dists[iss_i] = np.array([[]]).T
    return neighs, dists


def _array_autoexclude(to_exclude_elements, neighs, dists):
    """Over-writable function to exclude the own elements from the retrieved
    set.
    * neighs {array form} (iss, nei)
    """
    neighs, dists = _list_autoexclude(to_exclude_elements, neighs, dists)
    ## Transformation to array
    neighs = np.array(neighs)
    if dists is not None:
        dists = np.array(dists)
    return neighs, dists


def _general_autoexclude(to_exclude_elements, neighs, dists):
    if type(neighs) == np.ndarray:
        neighs, dists = _array_autoexclude(to_exclude_elements, neighs, dists)
    elif type(neighs) == list:
        neighs, dists = _list_autoexclude(to_exclude_elements, neighs, dists)
    return neighs, dists
