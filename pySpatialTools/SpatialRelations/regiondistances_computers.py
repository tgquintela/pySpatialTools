
"""
RegionDistances computers
-------------------------
Functions to compute in a different ways the distances between collections of
elements (or regions).

"""

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist

from pySpatialTools.Retrieve import\
    _discretization_regionlocs_parsing_creation
#from aux_regionmetrics import get_regions4distances
#    create_sp_descriptor_points_regs #, create_sp_descriptor_regionlocs
from pySpatialTools.FeatureManagement import _spdesc_parsing_creation
from pySpatialTools.FeatureManagement.features_objects import PhantomFeatures


def compute_ContiguityRegionDistances(discretizor, store='network'):
    """Region distances defined only by a contiguity measure defined in the
    discretization method. Function to compute the spatial distances between
    the regions.
    WARNING: Depends on contiguity (not defined in most of the retrievers

    Parameters
    ----------
    discretizor: pst.Discretization object
        the discretization information to transform locations into regions.
    store: optional, ['network', 'sparse', 'matrix']
        the type of object we want to store the relation metric.

    Returns
    -------
    relations: networkx, scipy.sparse, np.ndarray
        the relationship information between regions.
    _data: np.ndarray
        the regions_id.
    symmetric: boolean
        if the relations are symmetric or not (in order to save memory space).
    store: str
        how we want to store the relations.
    """
    ## 0. Preparing variables
    #_data = discretizor.reshape((discretizor.shape[0], 1))
    _data = discretizor.get_regions_id()

    ## 1. Computation of relations
    relations = discretizor.get_contiguity()

    ## 2. Formatting output
    symmetric = np.all((relations.T - relations).A)
    if store == 'matrix':
        relations = relations.A
    elif store == 'sparse':
        pass
    elif store == 'network':
        relations = nx.from_scipy_sparse_matrix(relations)
        mapping = dict(zip(relations.nodes(), _data))
        relations = nx.relabel_nodes(relations, mapping)
    pars_rel = {'symmetric': symmetric, 'store': store}
    return relations, pars_rel, _data


def compute_CenterLocsRegionDistances(sp_descriptor, store='network',
                                      elements=None, symmetric=False,
                                      activated=None):
    """Region distances defined only by the distance of the representative
    points of each region.
    It can be cutted by a definition of a retriever neighbourhood of each
    representative point.
    This function to compute the spatial distances between the regions.

    Parameters
    ----------
    TODO:
    sp_descriptor: sp_descriptor or tuple.
        the spatial descriptormodel object or the tuple of elements needed
        to build it (discretizor, locs, retriever, descriptormodel)
    store: optional, ['network', 'sparse', 'matrix']
        the type of object we want to store the relation metric.
    elements: array_like, list or None
        the regions we want to use for measure the metric distance between
        them.
    symmetric: boolean
        assume symmetric measure.
    activated: numpy.ndarray or None
        the location points we want to know if we use the regions non-empty
        or None if we want to use all of them.

    Returns
    -------
    relations: networkx, scipy.sparse, np.ndarray
        the relationship information between regions.
    _data: np.ndarray
        the regions_id.
    symmetric: boolean
        if the relations are symmetric or not (in order to save memory space).
    store: str
        how we want to store the relations.

    """
    ## 0. Preparing variables
    pars_rel = {'distanceorweighs': True, 'symmetric': symmetric}

    ## 1. Computing
    if type(sp_descriptor) == tuple:
        disc_info, retriever_info, _ = sp_descriptor
        centerlocs, regs =\
            _discretization_regionlocs_parsing_creation(disc_info, elements,
                                                        activated)
        if retriever_info is None:
            relations = cdist(centerlocs, centerlocs)
            pars_rel['symmetric'] = True
        else:
            assert(type(retriever_info) == tuple)
            retriever_info = tuple([retriever_info[0]] + [centerlocs] +
                                   list(retriever_info[1:]))

            ## TODO:
            # Instantiate PhantomFeatures
            feats_info = PhantomFeatures((None, len(regs)), characterizer=TODO)
            # Map_vals_i (not sure if it is needed)
            #pars_feats['map_vals_i'] = Map_vals_i(regs)
            sp_descriptor = _spdesc_parsing_creation(retriever_info,
                                                     feats_info)
            relations = sp_descriptor.compute_net()[:, :, 0]
    else:
        relations = sp_descriptor.compute_net()[:, :, 0]
        regs = np.arange(len(relations))

    ## 2. Formatting output
    if store == 'matrix':
        pass
    elif store == 'sparse':
        relations = coo_matrix(relations)
    elif store == 'network':
        relations = coo_matrix(relations)
        relations = nx.from_scipy_sparse_matrix(relations)
        mapping = dict(zip(relations.nodes(), regs))
        relations = nx.relabel_nodes(relations, mapping)
    _data = regs
    return relations, pars_rel, _data


def compute_AvgDistanceRegions(sp_descriptor, store='network', elements=None,
                               symmetric=False, activated=None):
    """Average distance of points of the different regions.
    Function to compute the spatial distances between regions.

    Parameters
    ----------
    TODO:
    locs: np.ndarray
        the locations of the elements.
    discretizor: pst.Discretization object
        the discretization information to transform locations into regions.

    store: optional, ['network', 'sparse', 'matrix']
        the type of object we want to store the relation metric.
    elements: array_like, list or None
        the regions we want to use for measure the metric distance between
        them.
    activated: numpy.ndarray or None
        the location points we want to know if we use the regions non-empty
        or None if we want to use all of them.

    Returns
    -------
    relations: networkx, scipy.sparse, np.ndarray
        the relationship information between regions.
    _data: np.ndarray
        the regions_id.
    symmetric: boolean
        if the relations are symmetric or not (in order to save memory space).
    store: str
        how we want to store the relations.

    """
    pars_rel = {'distanceorweighs': False, 'symmetric': symmetric}
    ## 1. Computing
    if type(sp_descriptor) == tuple:
        disc_info, retriever_info, _ = sp_descriptor
        centerlocs, _data = disc_info
        assert(retriever_info is not None)
        ## TODO:
        feats_info = PhantomFeatures((None, len(_data)), characterizer=TODO)
        ########
        sp_descriptor = _spdesc_parsing_creation(retriever_info, feats_info)
        relations = sp_descriptor.compute_net()[:, :, 0]
    else:
        relations = sp_descriptor.compute_net()[:, :, 0]
        _data = np.arange(len(relations))
    ## 2. Formatting output
    if store == 'matrix':
        pass
    elif store == 'sparse':
        relations = coo_matrix(relations)
    elif store == 'network':
        relations = coo_matrix(relations)
        relations = nx.from_scipy_sparse_matrix(relations)
        mapping = dict(zip(relations.nodes(), _data.ravel()))
        relations = nx.relabel_nodes(relations, mapping)
    pars_rel = {'symmetric': symmetric, 'store': store}

    return relations, pars_rel, _data


def compute_PointsNeighsIntersection(sp_descriptor, store='network',
                                     elements=None, symmetric=False,
                                     activated=None):
    """Region distances defined only by the intersection of neighbourhoods
    of the points belonged to each region.
    It is also applyable for the average distance between each points.
    Function to compute the spatial distances between the regions.

    Parameters
    ----------
    sp_descriptor: sp_descriptor or tuple.
        the spatial descriptormodel object or the tuple of elements needed
        to build it (discretizor, locs, retriever, descriptormodel)
    store: optional, ['network', 'sparse', 'matrix']
        the type of object we want to store the relation metric.
    elements: array_like, list or None
        the regions we want to use for measure the metric distance between
        them.
    symmetric: boolean
        assume symmetric measure.
    activated: numpy.ndarray or None
        the location points we want to know if we use the regions non-empty
        or None if we want to use all of them.

    Returns
    -------
    relations: networkx, scipy.sparse, np.ndarray
        the relationship information between regions.
    _data: np.ndarray
        the regions_id.
    symmetric: boolean
        if the relations are symmetric or not (in order to save memory space).
    store: str
        how we want to store the relations.

    """

    pars_rel = {'distanceorweighs': False, 'symmetric': symmetric}
    ## 1. Computing
    if type(sp_descriptor) == tuple:
        disc_info, retriever_info, _ = sp_descriptor
        assert(retriever_info is not None)
        centerlocs, regs = disc_info
        _data = np.unique(regs)
        ## TODO:
        feats_info = PhantomFeatures((None, len(_data)), characterizer=TODO)
        ########
        sp_descriptor = _spdesc_parsing_creation(retriever_info, feats_info)
        relations = sp_descriptor.compute_net()[:, :, 0]
    else:
        relations = sp_descriptor.compute_net()[:, :, 0]
        _data = np.arange(len(relations))
    ## 2. Formatting output
    if store == 'matrix':
        pass
    elif store == 'sparse':
        relations = coo_matrix(relations)
    elif store == 'network':
        relations = coo_matrix(relations)
        relations = nx.from_scipy_sparse_matrix(relations)
        mapping = dict(zip(relations.nodes(), _data.ravel()))
        relations = nx.relabel_nodes(relations, mapping)
    pars_rel = {'symmetric': symmetric, 'store': store}

    return relations, pars_rel, _data
