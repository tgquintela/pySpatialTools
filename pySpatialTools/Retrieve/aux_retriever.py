
"""
auxiliar retriever
------------------
Auxialiar functions for retrieving.

"""

import numpy as np


class NullCoreRetriever:
    """Dummy null retriever container. It gives the structure desired by the
    retrievers classes to work properly.
    """

    def __init__(self, data):
        self.data = data


def _check_retriever(retriever):
    """The checker function to assert that the retriever object input is well
    formatted for this package.

    Parameters
    ----------
    retriever: pst.BaseRetriever
        the retriever object we want to check.

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


############################## Exclude functions ##############################
###############################################################################
def _list_autoexclude(to_exclude_elements, neighs, dists):
    """Over-writable function to exclude the own elements from the retrieved
    set.

    Parameters
    ----------
    to_exclude_elements: list
        the list of the elements we want to exclude.
    neighs: list [iss][nei]
        the neighbour id for each element.
    dists: list [iss][nei][dim]
        the neighbour distance for each element.

    Returns
    -------
    neighs: list [iss][nei]
        the neighbour id for each element.
    dists: list [iss][nei][dim]
        the neighbour distance for each element.

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

    Parameters
    ----------
    to_exclude_elements: list
        the list of the elements we want to exclude.
    neighs: np.ndarray (iss, nei)
        the neighbour id for each element.
    dists: list [iss][nei][dim]
        the neighbour distance for each element.

    Returns
    -------
    neighs: np.ndarray [iss][nei]
        the neighbour id for each element.
    dists: list [iss][nei][dim] or np.ndarray (iss, nei, dim)
        the neighbour distance for each element.

    """
    neighs, dists = _list_autoexclude(to_exclude_elements, neighs, dists)
    ## Transformation to array
    neighs = np.array(neighs)
    if dists is not None:
        dists = np.array(dists)
    return neighs, dists


def _general_autoexclude(to_exclude_elements, neighs, dists):
    """General autoexclude function. No assumption about type of inputs.

    Parameters
    ----------
    to_exclude_elements: list
        the list of the elements we want to exclude.
    neighs: np.ndarray [iss][nei]
        the neighbour id for each element.
    dists: list [iss][nei][dim]
        the neighbour distance for each element.

    Returns
    -------
    neighs: list [iss][nei] or np.ndarray (iss, nei)
        the neighbour id for each element.
    dists: list [iss][nei][dim] or np.ndarray (iss, nei, dim)
        the neighbour distance for each element.

    """
    if type(neighs) == np.ndarray:
        neighs, dists = _array_autoexclude(to_exclude_elements, neighs, dists)
    elif type(neighs) == list:
        neighs, dists = _list_autoexclude(to_exclude_elements, neighs, dists)
    return neighs, dists
