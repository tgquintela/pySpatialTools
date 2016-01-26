
"""
Auxiliary regionmetrics
-----------------------
Auxiliary functions to complement the regionmetrics object.

"""

from scipy.sparse import coo_matrix
import networkx as nx
import numpy as np
from pySpatialTools.Feature_engineering import SpatialDescriptorModel


def create_sp_descriptor_points_regs(sp_descriptor, regions_id, elements_i):
    discretizor, locs, retriever, info_ret, descriptormodel = sp_descriptor
    retriever = retriever(locs, info_ret, ifdistance=True)
    loc_r = discretizor.discretize(locs)
    map_locs = dict(zip(regions_id, elements_i))
    r_locs = np.array([int(map_locs[r]) for r in loc_r])
    descriptormodel = descriptormodel(r_locs, sp_typemodel='correlation')
    sp_descriptor = SpatialDescriptorModel(retriever, descriptormodel)
    n_e = locs.shape[0]
    sp_descriptor.reindices = np.arange(n_e).reshape((n_e, 1))
    return sp_descriptor


def create_sp_descriptor_regionlocs(sp_descriptor, regions_id, elements_i):
    discretizor, locs, retriever, info_ret, descriptormodel = sp_descriptor
    if type(retriever) == str:
        regionslocs = discretizor.get_regionslocs()[elements_i, :]
        return regionslocs, retriever

    retriever = retriever(discretizor.get_regionslocs()[elements_i, :],
                          info_ret, ifdistance=True)
    descriptormodel = descriptormodel(np.array(elements_i),
                                      sp_typemodel='matrix')
    sp_descriptor = SpatialDescriptorModel(retriever, descriptormodel)
    n_e = len(elements_i)
    sp_descriptor.reindices = np.arange(n_e).reshape((n_e, 1))
    return sp_descriptor


def get_regions4distances(discretizor, elements=None, activated=None):
    """Get regions id to compute distance between them.

    Parameters
    ----------
    discretizor:
    elements:
    activated:

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
    retriever:
    element_labels:
    typeoutput: str, ['network', 'sparse', 'matrix']
        the type of ouput representation we want.
    symmetric: boolean
        if True the resultant distance information if forced to only give the
        upperpart of the distance-matrix in order to save memory.

    """
    lista = []
    for reg in list(element_labels):
        ## TODO: change locs
        neighs, dists = retriever.retrieve_neighs(reg, ifdistance=True)
        neighs, dists = filter_possible(element_labels, neighs, dists)
        neighs, dists = np.array(neighs), np.array(dists)
        aux = [(reg, neighs[i], dists[i]) for i in range(len(list(neighs)))]
        lista += aux
    ## Transformation to a sparse matrix
    element_labels = np.array(element_labels)
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
    lista = np.array(lista)
    dts, iss, jss = lista[:, 2], lista[:, 0], lista[:, 1]
    relations = coo_matrix((dts, (iss, jss)), shape=sh)
    return relations