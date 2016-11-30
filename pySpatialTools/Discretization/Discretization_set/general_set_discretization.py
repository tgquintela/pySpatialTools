
"""
general set discretization
--------------------------
Discretization set. Explicit relations between elements and their groups.

"""

import numpy as np
from scipy.sparse import issparse, coo_matrix
from ..spatialdiscretizer import BaseSpatialDiscretizor


class SetDiscretization(BaseSpatialDiscretizor):
    """Set discretization is mapping between a non-metric space and another
    topological space.
    """
    format_ = 'explicit'

    def __init__(self, membership, regionlocs=None, metric_f=None):
        """Constructor function.

        Parameters
        ----------
        membership: numpy.ndarray, list of lists, scipy.sparse or list of dicts
            the membership information. If the assignation of each element to
            a collection of elements is unique the membership can be
            represented as a numpy array, else it will be represented as a list
            of lists.
            It is stored as an unidimensional array if it is not multiple or
            in an scipy.sparse way if it is multiple.

        """
        ## Class parameters initialization
        self.multiple, self.metric, self.n_dim = None, None, None
        self._initialization()
        self.metric_f = metric_f
        self.metric = False if metric_f is None else True
        ## Format membership
        self._membership, (self._n, self.multiple, self._weighted,
                           self.regionlocs, self.regions_id) =\
            format_membership(membership)
        self.regions_id = self.regions_id.astype(int)
        if self.regionlocs is None:
            self.regionlocs = np.arange(self._n)
        self.multiple = not self.multiple

    @property
    def borders(self):
        return self._membership

    def _map_loc2regionid(self, elements):
        """Discretize locs returning their region_id.

        Parameters
        ----------
        elements: optional
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regions: numpy.ndarray
            the region_id of each locs for this discretization.
        ----
        if _weighted:
            regions: numpy.ndarray or list or lists
                the region_id of each locs for this discretization.
            weights: numpy.ndarray or list of lists
                the weights of membership to each region.

        """
        if not '__len__' in dir(elements):
            elements = [elements]
        inttypes = [int, np.int32, np.int64]
        ifindex = all([type(e) in inttypes for e in elements])
        if not ifindex and self.regionlocs is None:
            raise TypeError("Index needed.")
        ## Compute regions
        regions = -1*np.ones(len(elements)).astype(int)
        regions, weights = [], []
        for i in xrange(len(elements)):
            ## Getting indice of element
            if ifindex:
                j = elements[i]
            else:
                j = find_idx_in_array(elements[i], self.regionlocs)
            ## Getting relations
            if j >= 0 or j < self._n:
                regions_i, weights_i =\
                    indexing_rows(j, self._membership, self.regions_id,
                                  self.multiple)
                regions.append(regions_i)
                weights.append(weights_i)
            else:
                regions.append(np.array([-1]))
                weights.append([0])
        ## Formatting output
        if not self.multiple:
            regions, weights = np.array(regions), np.array(weights)
        if self._weighted:
            return regions, weights
        else:
            return regions

    def _map_regionid2regionlocs(self, regions=None):
        """Function which maps the regions ID to their elements.
        """
        regions = self.regions_id if regions is None else regions
        if type(regions) == int:
            regions = np.array([regions])
        elements, weights = [], []
        for i in xrange(len(regions)):
            ## Getting col indice
            idx = np.where(regions[i] == self.regions_id)[0]
            if self.multiple:
                j_col = idx[0]
            else:
                j_col = regions[i]
            ## Getting relations
            if len(idx):
                elements_i, weights_i =\
                    indexing_cols(j_col, self._membership,
                                  self.regionlocs, self.multiple)
                elements.append(elements_i)
                weights.append(weights_i)
            else:
                elements.append(np.array([-1]))
                weights.append([0])
        if self._weighted:
            return elements, weights
        else:
            return elements

    def _compute_limits(self, region_id=None):
        "Build the limits of the region discretized."
        if region_id is None:
            self.limits = np.array(0)
        else:
            return region_id

    def _compute_contiguity_geom(self, region_id, params={}):
        """Compute geometric contiguity."""
        if 'metric' in dir(self):
            if self.metric is not True:
                if self.metric_f is not None:
                    self.metric_f(self.regions_id, *params)
        raise Exception("Any metric defined.")


def format_membership(membership):
    """Format membership to fit it into the set discretizor standart."""
    if type(membership) == np.ndarray:
        _membership = membership
        n_elements = membership.shape[0]
        _unique = True
        _weighted = False
        collections = None
        collections_id = np.unique(membership)

    elif type(membership) == list:
        collections = None
        n_elements = len(membership)
        # Formatting to all list
        types = np.array([type(e) for e in membership])
        op1 = np.any([t in [np.ndarray, list] for t in types])
        op2 = np.all([t == dict for t in types])
        op30 = np.all([t == list for t in types])
        op31 = np.all([t == np.ndarray for t in types])
        op3 = op30 or op31
        if op1:
            for i in xrange(len(membership)):
                if type(membership[i]) not in [np.ndarray, list]:
                    membership[i] = [membership[i]]
            op3 = True
            types = np.array([type(e) for e in membership])
            op31 = np.all([t == np.ndarray for t in types])
        # Computing if dicts
        if op2:
            _membership = membership
            _unique = False
            _weighted = True
            n_elements = len(membership)
            aux = [membership[i].keys() for i in xrange(n_elements)]
            aux = np.hstack(aux)
            collections_id = np.unique(aux)
        # Computing if lists
        if op3:
            if op31:
                membership = [list(m) for m in membership]
            length = np.array([len(e) for e in membership])
            if np.all(length == 1):
                membership = np.array(membership)
                _membership = membership
                n_elements = membership.shape[0]
                _unique = True
                _weighted = False
                collections_id = np.unique(membership)
            else:
                _membership = membership
                n_elements = len(membership)
                _unique = False
                _weighted = False
                aux = np.hstack(membership)
                collections_id = np.unique(aux)
    elif issparse(membership):
        collections = None
        _membership = membership
        n_elements = membership.shape[0]
        collections_id = np.arange(membership.shape[1])
        _weighted = np.any(membership.data != 1)
        _unique = np.all(membership.sum(1) == 1)
    ## Transform to sparse
    out = n_elements, _unique, _weighted, collections, collections_id
    if not _unique:
        _membership, _ = to_sparse(_membership, out)
    return _membership, out


def to_sparse(_membership, out):
    "Return a sparse matrix object."
    n_elements, _unique, _weighted, collections, collections_id = out
    sh = n_elements, len(collections_id)
    aux_map = dict(zip(collections_id, range(sh[1])))
    if issparse(_membership):
        return _membership, (range(n_elements), collections_id)
    if _unique:
        _membership = np.array(_membership).ravel()
        matrix = np.array([aux_map[e] for e in _membership])
        matrix = matrix.astype(int)
        matrix = coo_matrix((np.ones(sh[0]), (range(sh[0]), matrix)),
                            shape=sh)
    elif not _weighted:
        indices = []
        for i in xrange(sh[0]):
            for j in range(len(_membership[i])):
                indices.append((i, aux_map[_membership[i][j]]))
        indices = np.array(indices)[:, 0], np.array(indices)[:, 1]
        matrix = coo_matrix((np.ones(len(indices[0])), indices), shape=sh)
    elif _weighted:
        indices, data = [], []
        for i in xrange(sh[0]):
            for j in _membership[i]:
                indices.append((i, aux_map[j]))
                data.append(_membership[i][j])
        indices = np.array(indices)[:, 0], np.array(indices)[:, 1]
        matrix = coo_matrix((np.array(data), indices), shape=sh)
    return matrix, (range(n_elements), collections_id)


def find_idx_in_array(element, elements):
    "Return the index of the first coincidence in an iterable."
    for i in xrange(len(elements)):
        if element == elements[i]:
            return i


def indexing_rows(j_row, _membership, regions_id, multiple):
    """Indexing rows from the membership relations."""
    if multiple:
        relations_i = _membership.getrow(j_row).A.ravel()
        idxs = np.where(relations_i)[0]
        regions_i, weights_i = regions_id[idxs], relations_i[idxs]
    else:
        regions_i = _membership[j_row]
        weights_i = np.array([1.])
    return regions_i, weights_i


def indexing_cols(j_col, _membership, regionlocs, multiple):
    """Indexing cols from the membership relations."""
    ## Getting elements and weights
    if multiple:
        relations_i = _membership.getcol(j_col).A
        idxs = relations_i.nonzero()
        idxs = idxs[0] if len(idxs) else idxs
        elements_i, weights_i = idxs, relations_i[idxs]
    else:
        idxs = np.where(_membership == j_col)[0]
        #idxs = idxs[0] if len(idxs) else idxs
        elements_i, weights_i = idxs, np.ones(len(idxs))
    ## Formatting to regionlocs
    if regionlocs is not None:
        elements_i = regionlocs[idxs]
    return elements_i, weights_i
