
"""

"""

from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np


def compute_selfdistances(retriever, element_labels, typeoutput='network',
                          symmetric=True):
    """Compute the self distances of the neighbourhood network defined by the
    retriever object.

    Parameters
    ----------
    retriever:
    element_labels:
    typeoutput: str, ['network', 'sparse', 'matrix']
        the type of ouput representation we want.
    symmetric: boolean
        if True the resultant distance information if forced to only give the
        upperpart of the distance-matrix in order to save memory.

    """
    lista = []
    for reg in element_labels:
        ## TODO: change locs
        neighs, dists = retriever.retrieve_neighs(reg, ifdistance=True)
        neighs, dists = filter_possible(element_labels, neighs, dists)
        aux = [(reg, neighs[i], dists[i]) for i in range(len(list(neighs)))]
        lista.append(aux)
    ## Transformation to a sparse matrix
    relations = sparse_from_listaregneighs(lista, element_labels, symmetric)
    if typeoutput == 'network':
        relations = nx.from_scipy_sparse_matrix(relations)
    if typeoutput == 'matrix':
        relations = relations.A
    return relations


def filter_possible(only_possible, neighs, dists):
    """Filter the neighs and dists only to the possible neighs.

    Parameters
    ----------
    only_possible: list or array_like
        the possible tags for neighs.
    neighs: list or array_like
        the retrieved neighs.
    dists: list or array_like
        the distance to the retrieved neighs.

    Returns
    -------
    neighs: list or array_like
        the retrieved filtered neighs.
    dists: list or array_like
        the distance to the retrieved filtered neighs.
    """
    idxs_filtered = [i for i in range(len(list(neighs)))
                     if neighs[i] in only_possible]
    neighs = [neighs[i] for i in range(len(list(neighs)))
              if i in idxs_filtered]
    dists = [dists[i] for i in range(len(list(dists)))
             if i in idxs_filtered]
    return neighs, dists


def sparse_from_listaregneighs(lista, u_regs, symmetric):
    """Sparse representation matrix from a list of tuples of indices and
    values.
    """
    sh = (u_regs.shape[0], u_regs.shape[0])
    dts, iss, jss = [], [], []
    for i in xrange(len(lista)):
        n_neigh = lista[i][1].shape[0]
        for j in range(n_neigh):
            dts.append(lista[i][0])
            iss.append(lista[i][1][j])
            jss.append(lista[i][2][j])
            if symmetric:
                dts.append(lista[i][0])
                iss.append(lista[i][2][j])
                jss.append(lista[i][1][j])
    dts, iss, jss = np.array(dts), np.array(iss), np.array(jss)
    relations = coo_matrix((dts, (iss, jss)), shape=sh)
    return relations
