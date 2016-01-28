
"""
RegionMetrics
-------------
The definition of distances between regions and the store of this measures
into an object.

Extend this objects to different input output.
Create a superClass.

"""

import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix, issparse
from scipy.spatial.distance import pdist, cdist
from itertools import combinations_with_replacement

from aux_regionmetrics import get_regions4distances,\
    create_sp_descriptor_points_regs, create_sp_descriptor_regionlocs


###############################################################################
###############################################################################
###############################################################################
class RegionDistances:
    """Object which stores the information of spatial relations between regions
    defined by a discretization of a discretized points.
    """
    _distanceorweighs = True
    _null_value = np.inf
    _inv_null_value = 0.

    relations = None
    _data = None
    _data_input = None
    _store = 'matrix'  # sparse, network
    _out = 'indices'  # indices, elements_id
    _input = 'indices'  # indices, elements_id

    def __init__(self, relations=None, distanceorweighs=True, symmetric=True,
                 input_='indices', output='indices'):
        ## Relations management
        self.relations = relations
        if relations is not None:
            # Store associated data
            if issparse(relations):
                sh0 = relations.shape[0]
                self._data = np.arange(sh0).reshape((sh0, 1))
            else:
                self._data = np.arange(len(relations)).reshape((len(relations), 1))
            # Type of input
            if type(relations) == np.ndarray:
                self._store = 'matrix'
            elif type(relations) == nx.Graph:
                self._store = 'network'
            elif issparse(relations):
                self._store = 'sparse'
        ## Type of values
        self._distanceorweighs = distanceorweighs
        if not distanceorweighs:
            self._null_value = 0.
            self._inv_null_value = np.inf
        self._symmetric = symmetric
        ## IO parameters
        self._input = input_
        self._out = output

    @property
    def data(self):
        return self._data

    def retrieve_neighs(self, reg):
        """Retrieve the neighbourhood regions of the region in input.

        Parameters
        ----------
        reg: int or numpy.ndarray
            the region_id which we want to retrieve their nieghborhood regions.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood points or regions.
        dists: numpy.ndarray
            the distances between points or regions.

        TODO:
        ----
        Errors when there is not in the data list.
        """
        ## 0. Format input
        if type(reg) == list:
            if len(reg) == 1:
                reg = reg[0]
                if reg not in self._data[:, 0]:
                    neighs = np.array([])
                    dists = np.array([])
                    return neighs, dists
            else:
                print reg, len(reg)
                raise TypeError("Not correct input.")
        elif type(reg) == np.ndarray:
            if len(reg.shape) == 1 and reg.shape[0] == 1:
                reg = int(reg[0])
            else:
                reg = int(reg)
            if type(reg) == int:
                if reg not in self._data[:, 0]:
                    neighs = np.array([])
                    dists = np.array([])
                    return neighs, dists
            else:
                print reg, reg.shape
                raise Exception("Not correct input.")
        elif type(reg) == int:
            reg = self._data[reg, 0]
        ## 1. Perform the retrieve
        if self.relations is not None:
            if self._store == 'matrix':
                logi = self.relations[self._data == reg, :] != self.null_value
                logi = logi[:, 0]
                if self._out == 'elements_id':
                    neighs = self._data[logi, 0]
                else:
                    neighs = np.where(logi)[0]
                dists = self.relations[self._data[:, 0] == reg, logi]
            elif self._store == 'sparse':
                i_reg = np.where(self._data[:, 0] == reg)[0][0]
                idxs = self.relations.getrow(i_reg).nonzero()[1]
                if self._symmetric:
                    idxs2 = self.relations.getcol(i_reg).nonzero()[0]
                    idxs = np.unique(np.hstack([idxs, idxs2]))
                dists = [self.relations.getrow(i_reg).getcol(i).A[0, 0]
                         for i in idxs]
                if self._out == 'elements_id':
                    neighs = self._data[idxs, 0]
                else:
                    neighs = idxs
            elif self._store == 'network':
                neighs = self.relations.neighbors(reg)
                dists = [self.relations[reg][nei]['weight'] for nei in neighs]
        else:
            neighs, dists = self.get_relations_spec(reg)
        neighs, dists = np.array(neighs).ravel(), np.array(dists)
        return neighs, dists

    def transform(self, f_trans, params={}):
        ##TODO:
        return f_trans(self.relations)

    @property
    def data_input(self):
        if self._data_input is None:
            return self._data
        else:
            return self.data_input

    @property
    def data_output(self):
        return self._data

    def __getitem__(self, i):
        if type(i) == list:
            neighs, dists = [], []
            for j in i:
                aux = self[j]
                neighs.append(aux[0])
                dists.append(aux[0])
        if type(i) == int:
            if self.shape[0] <= i or i < 0:
                raise IndexError('Index i out of bounds.')
            neighs, dists = self.retrieve_neighs(i)
        if type(i) == np.ndarray:
            neighs, dists = self.retrieve_neighs(i)
        if isinstance(i, slice):
            neighs, dists = self[list(range(i.start, i.stop, i.step))]
        if type(i) not in [int, list, slice, np.ndarray]:
            raise TypeError("Not correct index")
        return neighs, dists

    @property
    def shape(self):
        return (len(self.data_input), len(self.data_output))


class CenterLocsRegionDistances(RegionDistances):
    """Region distances defined only by the distance of the representative
    points of each region.
    It can be cutted by a definition of a retriever neighbourhood of each
    representative point.
    """

    def compute_distances(self, sp_descriptor, store='network', elements=None,
                          symmetric=True, activated=None):
        """Function to compute the spatial distances between the regions.

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
        """
        ## 0. Compute variable needed
        self._symmetric = symmetric
        self._store = store
        # Sp descriptor management
        if type(sp_descriptor) == tuple:
            activated = sp_descriptor[1] if activated is not None else None
            regions_id, elements_i = get_regions4distances(sp_descriptor[0],
                                                           elements, activated)
            sp_descriptor = create_sp_descriptor_regionlocs(sp_descriptor,
                                                            regions_id,
                                                            elements_i)
            self._data = np.array(regions_id)
            self._data = self._data.reshape((self._data.shape[0], 1))
        else:
            regions, elements_i = get_regions4distances(sp_descriptor,
                                                        elements, activated)
            self._data = np.array(regions)
            self._data = self._data.reshape((self._data.shape[0], 1))

        ## 1. Computation of relations
        if type(sp_descriptor) == tuple:
            relations = pdist(sp_descriptor[0], sp_descriptor[1])
        else:
            relations = sp_descriptor.compute_net()[:, :, 0]
        if store == 'matrix':
            self.relations = relations
        elif store == 'sparse':
            self.relations = coo_matrix(relations)
        elif store == 'network':
            relations = coo_matrix(relations)
            self.relations = nx.from_scipy_sparse_matrix(relations)
            mapping = dict(zip(self.relations.nodes(), regions_id))
            self.relations = nx.relabel_nodes(self.relations, mapping)


class ContiguityRegionDistances(RegionDistances):
    """Region distances defined only by a contiguity measure defined in the
    discretization method.
    """

    def compute_distances(self, discretizor, store='network'):
        """Function to compute the spatial distances between the regions.
        """
        ## TODO: implement contiguity into the discretizor
        self._data = discretizor
        self._data = self._data.reshape((self._data.shape[0], 1))
        self.relations = discretizor.retrieve_contiguity_regions(store)


class AvgDistanceRegions(RegionDistances):
    """Average distance of points of the different regions.
    """

    def compute_distances(self, locs, discretizor, regretriever,
                          store='network', elements=None, activated=None):
        """Function to compute the spatial distances between regions.

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
        """
        self._symmetric = True
        self._store = store

        regs = discretizor.discretize(locs)
        u_regs = np.unique(regs)
        n_regs = len(u_regs)
        self._data = u_regs.reshape((len(u_regs), 1))

        dts, iss, jss = [], [], []
        for i in xrange(len(u_regs)):
            locs_i = locs[regs == u_regs[i]]
            neighs_i, _ = regretriever.retrieve_neighs(u_regs[i])
            for j in range(len(neighs_i)):
                locs_j = locs[regs == neighs_i[j]]
                dists_j = np.where(neighs_i[j] == u_regs)[0]
                if len(locs_j):
                    dts.append(cdist(locs_i, locs_j).mean())
                    iss.append(i)
                    jss.append(dists_j[0])

        dts, iss, jss = np.hstack(dts), np.hstack(iss), np.hstack(jss)
        iss, jss = iss.astype(int), jss.astype(int)
        relations = coo_matrix((dts, (iss, jss)), shape=(n_regs, n_regs))

        if store == 'matrix':
            self.relations = relations
        elif store == 'sparse':
            self.relations = coo_matrix(relations)
        elif store == 'network':
            relations = coo_matrix(relations)
            self.relations = nx.from_scipy_sparse_matrix(relations)
            mapping = dict(zip(self.relations.nodes(), u_regs))
            self.relations = nx.relabel_nodes(self.relations, mapping)


class PointsNeighsIntersection(RegionDistances):
    """Region distances defined only by the intersection of neighbourhoods
    of the points belonged to each region.
    It is also applyable for the average distance between each points.
    """

    def compute_distances(self, sp_descriptor, store='network', elements=None,
                          symmetric=False, activated=None):
        """Function to compute the spatial distances between the regions.

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
            self._data = np.array(regions_id)
            self._data = self._data.reshape((self._data.shape[0], 1))
        else:
            regions, elements_i = get_regions4distances(sp_descriptor,
                                                        elements, activated)
            self._data = np.array(regions)
            self._data = self._data.reshape((self._data.shape[0], 1))

        self._symmetric = symmetric
        self._store = store
        ## 1. Computation of relations
        relations = sp_descriptor.compute_net()[:, :, 0]
        #filter_relations(relations, self.data, elements)
        if store == 'matrix':
            self.relations = relations
        elif store == 'sparse':
            self.relations = coo_matrix(relations)
        elif store == 'network':
            relations = coo_matrix(relations)
            self.relations = nx.from_scipy_sparse_matrix(relations)
            mapping = dict(zip(self.relations.nodes(), regions_id))
            self.relations = nx.relabel_nodes(self.relations, mapping)
