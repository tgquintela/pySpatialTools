
"""
Auxiliary regionmetrics
-----------------------
Auxiliary functions to complement the regionmetrics object.

"""

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix


def get_regions4distances(discretizor, elements=None, activated=None):
    """Get regions id to compute distance between them.

    Parameters
    ----------
    discretizor: pst.BaseDiscretizor
        the discretization information.
    elements: np.ndarray or None (default)
        the elements we want to consider. If None, all are considered.
    activated: boolean or None (default)
        the use only of the regions which have elements in them or all.

    Returns
    -------
    regions_id:
    elements_i: list of int
        the list of indices of the regions we will use.
    """
    if activated is None:
        regions_id = discretizor.get_regions_id()
        if elements is not None:
            regions_id = elements
            elements_i = [int(np.where(regions_id == e)[0])
                          for e in regions_id]
        else:
            elements_i = range(regions_id.shape[0])
    else:
        regions_id = discretizor.discretize(activated)
        regions_id = np.unique(regions_id)
        elements_i = [int(np.where(regions_id == e)[0]) for e in regions_id]
    return regions_id, elements_i


def compute_selfdistances(retriever, element_labels, typeoutput='network',
                          symmetric=True):
    """Compute the self distances of the neighbourhood network defined by the
    retriever object.

    Parameters
    ----------
    retriever: pst.BaseRetriever
        the retriever object.
    element_labels: np.ndarray
        the element labels.
    typeoutput: str, ['network', 'sparse', 'matrix']
        the type of ouput representation we want.
    symmetric: boolean
        if True the resultant distance information if forced to only give the
        upperpart of the distance-matrix in order to save memory.

    """
    lista = []
    for reg in list(element_labels):
        neighs_info = retriever.retrieve_neighs(reg, ifdistance=True)
        # Neighs_info management
        neighs, dists, _, _ = neighs_info.get_information(0)
        neighs, dists = neighs[0], dists[0]
        neighs, dists = filter_possible_neighs(element_labels, neighs, dists)
        neighs, dists = np.array(neighs[0]), np.array(dists[0])
        aux = [(reg, neighs[i], dists[i]) for i in range(len(list(neighs)))]
        lista += aux
    ## Transformation to a sparse matrix
    element_labels = np.array(element_labels)
    relations = sparse_from_listaregneighs(lista, element_labels, symmetric)
    if typeoutput == 'network':
        relations = nx.from_scipy_sparse_matrix(relations)
    elif typeoutput == 'matrix':
        relations = relations.A
    return relations


def filter_possible_neighs(only_possible, neighs, dists):
    """Filter the neighs and dists only to the possible neighs.

    Parameters
    ----------
    only_possible: list or array_like
        the possible tags for neighs.
    neighs: list or array_like, [iss][nei]
        the retrieved neighs.
    dists: list or array_like, [iss][nei]
        the distance to the retrieved neighs.

    Returns
    -------
    neighs: list or array_like, [iss][nei]
        the retrieved filtered neighs.
    dists: list or array_like, [iss][nei]
        the distance to the retrieved filtered neighs.
    """

    for iss_i in range(len(neighs)):
        idxs_filtered = [i for i in range(len(list(neighs[iss_i])))
                         if neighs[iss_i][i] in only_possible]
        aux_nei, aux_dists = [], []
        for nei_i in range(len(list(neighs[iss_i]))):
            if nei_i in idxs_filtered:
                aux_nei.append(neighs[iss_i][nei_i])
                if dists is not None:
                    aux_dists.append(dists[iss_i][nei_i])

        neighs[iss_i] = np.array(aux_nei).astype(int)
        if dists is not None:
            dists[iss_i] = np.array(aux_dists)
    return neighs, dists


def sparse_from_listaregneighs(lista, u_regs, symmetric):
    """Sparse representation matrix from a list of tuples of indices and
    values.

    Parameters
    ----------
    lista: list
        the list of relations.
    u_regs: np.ndarray
        the regions id.
    symmetric: boolean
        if symmetric True else False.

    Returns
    -------
    coo_matrix: scipy.coo_matrix
        the scipy sparse coordinate representation of the relations.

    """
    sh = (u_regs.shape[0], u_regs.shape[0])
    lista = np.array(lista)
    dts, iss, jss = lista[:, 2], lista[:, 0], lista[:, 1]
    relations = coo_matrix((dts, (iss, jss)), shape=sh)
    return relations
