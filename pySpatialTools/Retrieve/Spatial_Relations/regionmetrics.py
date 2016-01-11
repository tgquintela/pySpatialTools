
"""
RegionMetrics
-------------
The definition of distances between regions and the store of this measures
into an object.

"""

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, issparse
from scipy.spatial.distance import pdist

from aux_regionmetrics import compute_selfdistances


###############################################################################
###############################################################################
###############################################################################
class RegionDistances:
    """Object which stores the information of spatial relations between regions
    defined by a discretization of a discretized points.
    """
    distanceorweighs = True
    null_value = np.inf
    inv_null_value = 0.

    relations = None
    u_regs = None
    store = 'matrix'  # sparse, network

    def __init__(self, relations=None, distanceorweighs=True, symmetric=True):
        ## Relations management
        self.relations = relations
        if type(relations) == np.ndarray:
            self.store = 'matrix'
        elif type(relations) == nx.Graph:
            self.store = 'network'
        elif issparse(relations):
            self.store = 'sparse'
        ## Type of values
        self.distanceorweighs = distanceorweighs
        if not distanceorweighs:
            self.null_value = 0.
            self.inv_null_value = np.inf
        self.symmetric = symmetric

    def get_relations(self, reg):
        """Retrieve the neighbourhood regions of the region in input.

        Parameters
        ----------
        reg: int
            the region_id which we want to retrieve their nieghborhood regions.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood points or regions.
        dists: numpy.ndarray
            the distances between points or regions.

        """
        if self.relations is not None:
            if self.store == 'matrix':
                logi = self.relations[self.u_regs == reg, :] != self.null_value
                neighs = self.u_reg[logi]
                dists = self.relations[self.u_regs == reg, logi]
            elif self.store == 'sparse':
                i_reg = np.where(self.u_regs == reg)[0][0]
                idxs = self.relations.getrow(i_reg).nonzero()[1]
                if self.symmetric:
                    idxs2 = self.relations.getcol(i_reg).nonzero()[0]
                    idxs = np.unique(np.hstack([idxs, idxs2]))
                dists = [self.relations.getrow(i_reg).getcol(i).A[0, 0]
                         for i in idxs]
                neighs = self.u_regs[idxs]
                self.relations[i_reg, idxs]
            elif self.store == 'network':
                neighs = self.relations.neighbors(reg)
                dists = [self.relations[reg][nei]['weight'] for nei in neighs]
        else:
            neighs, dists = self.get_relations_spec(reg)
        neighs, dists = np.array(neighs), np.array(dists)
        return neighs, dists


class CenterLocsRegionDistances(RegionDistances):
    """Region distances defined only by the distance of the representative
    points of each region.
    It can be cutted by a definition of a retriever neighbourhood of each
    representative point.
    """

    def compute_distances(self, discretizor, retriever='', store='network',
                          symmetric=True, elements=None):
        """Function to compute the spatial distances between the regions.
        TODO: spatial_descriptor
        """
        ## 0. Elements management
        if elements is not None:
            elements = discretizor.regions_id
        elif elements is True:
            # Filter by activated regions
            elements = np.unique(self.discretizor.discretize(retriever.data))
        self.u_regs = elements
        self.symmetric = symmetric

        ## 1. Computation of relations
        if type(retriever) == str:
            regionlocs = np.array(discretizor.regionlocs)
            metric = retriever if retriever else 'euclidean'
            self.relations = pdist(regionlocs, metric)
        else:
            # compute distances between to their neighs
            self.relations = compute_selfdistances(retriever, self.u_regs,
                                                   store, symmetric)


class ContiguityRegionDistances(RegionDistances):
    """Region distances defined only by a contiguity measure defined in the
    discretization method.
    """

    def compute_distances(self, discretizor, store='network'):
        """Function to compute the spatial distances between the regions.
        """
        ## TODO: implement contiguity into the discretizor
        self.u_regs = discretizor
        self.relations = discretizor.retrieve_contiguity_regions(store)


class PointsNeighsIntersection(RegionDistances):
    """Region distances defined only by the intersection of neighbourhoods
    of the points belonged to each region.
    """

    def compute_distances(self, sp_descriptor, store='network',
                          symmetric=False, geom=False, elements=None):
        """Function to compute the spatial distances between the regions.
        """
        ## 0. Needed variables
        # Filter by activated regions
        # Internal function if it is possible
        locs = sp_descriptor.retriever.retriever.data
        regionlocs, self.u_regs =\
            self.sp_descriptor.discretizor.get_activated_regionlocs(locs, geom)
        self.u_regs = np.unique(self.sp_descriptor.discretizor.discretize(self.sp_descriptor.retriever.data))

        self.symmetric = symmetric

        ## 1. Computation of relations
        relations, self.u_regs = sp_descriptor.compute_net()
        filter_relations(relations, self.u_regs, elements)
        if store == 'matrix':
            self.relations = relations
        elif store == 'sparse':
            self.relations = coo_matrix(relations)
        elif store == 'network':
            self.relations = coo_matrix(relations)
            self.relations = nx.from_scipy_sparse_matrix(relations)





    def compute_contiguity(self, retriever, locs, info_i):
        """Compute contiguity using the locations and a retriever.

        TODO
        ----
        Use correlation measure!!!!!
        """
        ## 0. Prepare inputs
        sh = locs.shape
        locs = locs if len(sh) > 1 else locs.reshape((1, sh[0]))
        ret = retriever(locs)
        regions_u = self.regions_id.unique()
        n_reg_u = regions_u.shape[0], regions_u.shape[0]
        regions_counts = np.zeros(n_reg_u)
        region_coincidences = np.zeros((n_reg_u, n_reg_u))
        ## 1. Compute matrix of coincidences
        regions = self.discretize(locs)
        for i in xrange(locs.shape[0]):
            r = regions[i]
            i_r = np.where(regions_u == r)
            neighs, dist = ret.retrieve_neighs(locs[i, :], info_i[i], True)
            regs = regions[neighs]
            i_regs = np.array([rs == regions_u for rs in regs])
            weights = compute_weigths(regs, dist)
            region_coincidences[i_r, i_regs] += weights
        contiguity = compute_measure(region_coincidences, regions_counts)
        return contiguity
