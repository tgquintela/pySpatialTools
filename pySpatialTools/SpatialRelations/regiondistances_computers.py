
"""
RegionDistances computers
-------------------------
Functions to compute in a different ways the distances between collections of
elements (or regions).

"""

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, cdist

from aux_regionmetrics import get_regions4distances
#    create_sp_descriptor_points_regs #, create_sp_descriptor_regionlocs
from pySpatialTools.FeatureManagement import _spdesc_parsing_creation
from pySpatialTools.Retrieve import\
    _discretization_regionlocs_parsing_creation


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
                                      elements=None, symmetric=True,
                                      activated=None):
    """Region distances defined only by the distance of the representative
    points of each region.
    It can be cutted by a definition of a retriever neighbourhood of each
    representative point.
    This function to compute the spatial distances between the regions.

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
    ## 0. Preparing variables
    pars_rel = {'distanceorweighs': True, 'symmetric': symmetric,
                'store': store}

    ## 1. Computing
    if type(sp_descriptor) == tuple:
        disc_info, retriever_info, _ = sp_descriptor
        ## TODO: get_regions4distances, create_sp_descriptor_regionlocs
        centerlocs, regs =\
            _discretization_regionlocs_parsing_creation(disc_info, elements,
                                                        activated)
        if retriever_info is None:
            relations = pdist(centerlocs, centerlocs)
            pars_rel['symmetric'] = True
        else:
            if type(retriever_info) != tuple:
                retriever_info = retriever_info, centerlocs
            else:
                retriever_info = tuple([retriever_info[0]] + [centerlocs] +
                                       list(retriever_info[1:]))

            ## TODO:
            # Instantiate PhantomFeatures
            # Map_vals_i
            sp_descriptor = _spdesc_parsing_creation(retriever_info)
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


def compute_AvgDistanceRegions(locs, discretizor, regretriever,
                               store='network', elements=None, activated=None):
    """Average distance of points of the different regions.
    Function to compute the spatial distances between regions.

    Parameters
    ----------
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

    Assumptions
    -----------
    k=0, iss=0

    """


    ## 1. Computing
    if type(sp_descriptor) == tuple:
        disc_info, retriever_info, _ = sp_descriptor
        ## TODO: get_regions4distances, create_sp_descriptor_regionlocs
        centerlocs, regs = disc_info
        if retriever_info is None:
            relations = pdist(centerlocs, centerlocs)
            pars_rel['symmetric'] = True
        else:
            sp_descriptor = _spdesc_parsing_creation(retriever_info)
            relations = sp_descriptor.compute_net()[:, :, 0]
    else:
        relations = sp_descriptor.compute_net()[:, :, 0]
        _data = np.arange(len(relations))



#    symmetric = True
#    iss_i, ki = 0, 0
#
#    regs = discretizor.discretize(locs)
#    u_regs = np.unique(regs)
#    u_regs = u_regs.reshape((len(u_regs), 1))
#    n_regs = len(u_regs)
#    _data = u_regs.reshape((len(u_regs), 1))
#    dts, iss, jss = [], [], []
#    for i in xrange(len(u_regs)):
#        locs_i = locs[regs == u_regs[i]]
#        neighs_info = regretriever.retrieve_neighs(u_regs[i])
#        neighs_i = neighs_info.get_neighs([0])
#        print list(u_regs.ravel()), neighs_i
#        for j in range(len(neighs_i[iss_i][ki])):
#            locs_j = locs[regs == neighs_i[iss_i][ki][j]]
#            dists_j = cdist(locs_i, locs_j).mean()
#            dts.append(dists_j)
#            iss.append(i)
#            jss.append(list(u_regs.ravel()).index(int(neighs_i[iss_i][ki][j])))
#    dts, iss, jss = np.hstack(dts), np.hstack(iss), np.hstack(jss)
#    iss, jss = iss.astype(int), jss.astype(int)
#    relations = coo_matrix((dts, (iss, jss)), shape=(n_regs, n_regs))

    ## 2. Formatting output
    if store == 'matrix':
        pass
    elif store == 'sparse':
        relations = coo_matrix(relations)
    elif store == 'network':
        relations = coo_matrix(relations)
        relations = nx.from_scipy_sparse_matrix(relations)
        mapping = dict(zip(relations.nodes(), u_regs.ravel()))
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
    ## 0. Needed variables
    # Sp descriptor management
    if type(sp_descriptor) == tuple:
        activated = sp_descriptor[1] if activated is not None else None
        regions_id, elements_i = get_regions4distances(sp_descriptor[0],
                                                       elements, activated)
        sp_descriptor = create_sp_descriptor_points_regs(sp_descriptor,
                                                         regions_id,
                                                         elements_i)
        _data = np.array(regions_id)
        _data = _data.reshape((_data.shape[0], 1))
    else:
        regions, elements_i = get_regions4distances(sp_descriptor,
                                                    elements, activated)
        _data = np.array(regions)
        _data = _data.reshape((_data.shape[0], 1))

    ## 1. Computation of relations
    relations = sp_descriptor.compute_net()[:, :, 0]
    #filter_relations(relations, self.data, elements)
    if store == 'matrix':
        pass
    elif store == 'sparse':
        relations = coo_matrix(relations)
    elif store == 'network':
        relations = coo_matrix(relations)
        relations = nx.from_scipy_sparse_matrix(relations)
        mapping = dict(zip(relations.nodes(), regions_id))
        relations = nx.relabel_nodes(relations, mapping)
    pars_rel = {'symmetric': symmetric, 'store': store}

    return relations, pars_rel, _data
