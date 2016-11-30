
"""
Membership
----------
Module which contains the classes and functions needed to define membership
object.
Membership object is a map between elements and collections to which they
belong. It is done to unify and simplify tasks.


"""

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix, issparse


class Membership:
    """Class representing a membership object in which maps every element
    assigned by an index to a group or collection of elements.
    Represent a mapping between elements and collections.

    """
    def _initialization(self):
        self._membership = []
        self.n_elements = 0
        self._unique = True
        self._weighted = False
        self.n_collections = 0
        self._maxcollection_id = -1

    def __init__(self, membership):
        """Constructor function.

        Parameters
        ----------
        membership: numpy.ndarray, list of lists or list of dicts
            the membership information. If the assignation of each element to
            a collection of elements is unique the membership can be
            represented as a numpy array, else it will be represented as a list
            of lists.

        """
        self._initialization()
        membership, out = format_membership(membership)
        self._membership = membership
        self._unique = out[1]
        self._weighted = out[2]
        self.collections_id = out[3]
        self.n_elements = out[0]
        self.n_collections = len(out[3])

    def __getitem__(self, i):
        """Returns the collections to which belong the element i.

        Parameters
        ----------
        i: int
            the element id we want to retrieve the collections to which it
            belongs.

        Returns
        -------
        collections: list
            the collections list which belong to the element i.

        """
        if i < 0 or i >= self.n_elements:
            raise IndexError("Element ID out of range")
        if issparse(self._membership):
            try:
                #irow = np.where(i == self.elements_id)[0][0]
                irow = i
            except:
                raise IndexError("Collection ID out of range")
            collections = np.zeros(self.n_elements).astype(bool)
            collections[self._membership.getrow(irow).nonzero()[0]] = True
            return collections
        if self._unique:
            collections = [self._membership[i]]
        else:
            if not self._weighted:
                collections = self._membership[i]
            else:
                collections = self._membership[i].keys()
        return collections

    def __eq__(self, collection_id):
        """The equal function collection by collection.

        Parameters
        ----------
        collection_id: int
            the collection code we want to obtain their linked elements.

        Returns
        -------
        elements: list or np.ndarray
            the elements associated to a collection given in the input.

        """
        elements = self.getcollection(collection_id)
        return elements

    def __iter__(self):
        """Iterates over the elements and return element i, its collections to
        which it belongs and the membership value.

        Returns
        -------
        i: int
            the element considered.
        colls: int or list or np.ndarray
            the collections associated to the value of the element `i`.
        memb_vals: float
            the membership value.

        """
        if issparse(self._membership):
            for i in xrange(self.n_elements):
                yield i, [self[i]], np.array([1.])
        elif self._unique:
            for i in xrange(self.n_elements):
                yield i, [self._membership[i]], np.array([1.])
        else:
            if not self._weighted:
                for i in xrange(self.n_elements):
                    colls = self._membership[i]
                    memb_vals = np.ones(len(colls))
                    yield i, colls, memb_vals
            else:
                for i in xrange(self.n_elements):
                    colls = self._membership[i].keys()
                    memb_vals = np.array(self._membership[i].values())
                    yield i, colls, memb_vals

    def __str__(self):
        """Return a representation of information about the data of this class.

        Returns
        -------
        summary: str
            the summary information of that class.

        """
        summary = """Membership class with %s elements and %s collections,
            in which there are %sweighted and %sunique relations between
            elements and collections."""
        wei = "" if self._weighted else "non-"
        unique = "" if self._unique else "non-"
        summary = summary % (self.n_elements, self.n_collections, wei, unique)
        return summary

    @property
    def membership(self):
        """Returns the membership data.

        Parameters
        ----------
        _membership: list, dict or np.ndarray
            the membership information.

        """
        return self._membership

    @property
    def shape(self):
        """The size of the inputs and outputs.

        Returns
        -------
        n_elements: int
            the number of elements.
        n_collections: int
            the number of collections.

        """
        return self.n_elements, self.n_collections

    @property
    def elements_id(self):
        """The element indices.

        Returns
        -------
        elements_id: list
            the elements codes.

        """
        return list(range(self.shape[0]))

    @property
    def max_collection_id(self):
        """

        Returns
        -------
        max_collection_id: int
            the number of maximum code of the collections.

        """
        try:
            return np.max(self.collections_id)
        except:
            maxs_id = [np.max(e) for e in self._membership if e]
            return np.max(maxs_id)

    def getcollection(self, collection_id):
        """Returns the members of the specified collection.

        Parameters
        ----------
        collection_id: int
            the collection id we want to retrieve its members.

        Returns
        -------
        elements: boolean numpy.ndarray
            the elements which belong to the collection_id.

        """
        if collection_id < 0 or collection_id > self.max_collection_id:
            raise IndexError("Collection ID out of range")
        if issparse(self._membership):
            try:
                jcol = np.where(collection_id == self.collections_id)[0][0]
            except:
                raise IndexError("Collection ID out of range")
            elements = np.zeros(self.n_elements).astype(bool)
            elements[self._membership.getcol(jcol).nonzero()[0]] = True
            return elements

        if self._unique:
            elements = self._membership == collection_id
        else:
            if not self._weighted:
                elements = [collection_id in self._membership[i]
                            for i in xrange(self.n_elements)]
                elements = np.array(elements).astype(bool)
            else:
                aux = [e.keys() for e in self._membership]
                elements = [collection_id in aux[i]
                            for i in xrange(self.n_elements)]
                elements = np.array(elements).astype(bool)
        return elements

    def to_network(self):
        """Return a networkx graph object.

        Returns
        -------
        membership: nx.Graph
            the network representation of the membership in a networkx package.
            The network has to be a bipartite network.

        """
        dict_relations = self.to_dict()
        G = nx.from_dict_of_dicts(dict_relations)
        return G

    def to_dict(self):
        """Return a dictionary object.

        Returns
        -------
        membership: dict
            the membership representation in a dictionary form.

        """
        element_lab = ["e %s" % str(e) for e in xrange(self.n_elements)]
        d = {}
        for i, collects_i, weighs in self:
            if np.sum(collects_i) == 0:
                collects_i = []
            else:
                collects_i = np.array(collects_i).ravel()
                collects_i = list(np.where(collects_i)[0])
            if self._weighted:
                collects_i = ["c %s" % str(c_i) for c_i in collects_i]
                weighs = [{'membership': w_i} for w_i in weighs]
                d[element_lab[i]] = dict(zip(collects_i, weighs))
            else:
                collects_i = ["c %s" % str(c_i) for c_i in collects_i]
                weighs = [{} for c_i in collects_i]
                d[element_lab[i]] = dict(zip(collects_i, weighs))
        return d

    def to_sparse(self):
        """Return a sparse object.

        Returns
        -------
        membership: scipy.sparse
            the membership representation in a sparse matrix representation.

        """
        au = self.n_elements, self._unique, self._weighted, self.collections_id
        return to_sparse(self._membership, au)

    def reverse_mapping(self):
        """Reverse the mapping of elements to collections to collections to
        elements.

        Returns
        -------
        reverse_membership: dict
            the reverse mapping between collections and elements.

        """
        reversed_ = {}
        for coll_id in self.collections_id:
            elements = self.getcollection(coll_id)
            elements = list(np.where(elements)[0])
            reversed_[coll_id] = elements
        return reversed_


def format_membership(membership):
    """Format membership to fit it into the set discretizor standart.

    Parameters
    ----------
    membership: dict, np.ndarray, list, scipy.sparse or nx.Graph
        the membership information.

    Returns
    -------
    membership: dict, np.ndarray, list, scipy.sparse or nx.Graph
        the membership information.

    """
    tuple2 = False
    # Pre compute
    if type(membership) == tuple:
        tuple2 = True
        collections_id2 = membership[1]
        membership = membership[0]
    # Computing if array
    if type(membership) == np.ndarray:
        _membership = membership
        n_elements = membership.shape[0]
        _unique = True
        _weighted = False
        collections_id = np.unique(membership)
    # Computing if list
    elif type(membership) == list:
        n_elements = len(membership)
        # Possibilities
        types = np.array([type(e) for e in membership])
        op1 = np.any([t == list for t in types])
        op2 = np.all([t == dict for t in types])
        op30 = np.all([t == list for t in types])
        op31 = np.all([t == np.ndarray for t in types])
        op3 = op30 or op31
        # Uniformation
        if op1:
            for i in xrange(len(membership)):
                if type(membership[i]) != list:
                    membership[i] = [membership[i]]
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
        elif op3:
            if op31:
                membership = [list(m) for m in membership]
            length = np.array([len(e) for e in membership])
            if np.all(length == 1):
                membership = np.array(membership).ravel()
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
        _membership = membership
        n_elements = membership.shape[0]
        collections_id = np.arange(membership.shape[1])
        _weighted = np.any(membership.data != 1)
        _unique = np.all(membership.sum(1) == 1)

    ## Final imputing
    if tuple2:
        collections_id = collections_id2
    ## Transform to sparse
    out = n_elements, _unique, _weighted, collections_id
#    if not _unique:
#        _membership, _ = to_sparse(_membership, out)
    return _membership, out


def to_sparse(_membership, out):
    """Return a sparse matrix object.

    Parameters
    ----------
    membership: dict, np.ndarray, list, scipy.sparse or nx.Graph
        the membership information.
    out: str, optional ['network', 'sparse', 'membership', 'list', 'matrix']
        the out format we want to output the membership.

    Returns
    -------
    membership: dict, np.ndarray, list, scipy.sparse or nx.Graph
        the membership information.

    """
    n_elements, _unique, _weighted, collections_id = out
    sh = n_elements, len(collections_id)
    aux_map = dict(zip(collections_id, range(sh[1])))
    if issparse(_membership):
        return _membership, (range(n_elements), collections_id)
    if _unique:
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


def formatrelationship(relationship, typeoutput):
    pass


"""
typeoutput = 'network', 'sparse', 'membership', 'list', 'matrix'
"""

#    def collections_id_computer():
#        "Return the collections ID of the membership."
#        # For each 3 cases
#        if self._unique:
#            collections_id = np.unique(self._membership)
#        elif not self._unique and not self._weighted:
#            collections_id = np.unique(np.hstack(self.membership)).astype(int)
#        elif not self._unique and self._weighted:
#            aux = [self._membership[e].keys() for e in self._membership]
#            collections_id = np.unique(np.hstack(aux))
#        return collections_id

#
#    def to_sparse(self):
#        "Return a sparse matrix object."
#        sh = self._nelements, self._ncollections
#        aux_map = dict(zip(self.collections_id, range(self._ncollections)))
#        if self._unique:
#            matrix = np.array([aux_map[e] for e in self._membership])
#            matrix = matrix.astype(int)
#            matrix = coo_matrix((np.ones(sh[0]), (range(sh[0], matrix))),
#                                shape=sh)
#        elif not self._weighted:
#            indices = []
#            for i in xrange(sh[0]):
#                for j in range(self._membership[i]):
#                    indices.append((i, aux_map[self._membership[i][j]]))
#            matrix = coo_matrix((np.ones(sh[0]), indices), shape=sh)
#
#        elif self._weighted:
#            indices, data = [], []
#            for i in xrange(sh[0]):
#                for j in self._membership[i]:
#                    indices.append((i, aux_map[j]))
#                    data.append(self._membership[i][j])
#            matrix = coo_matrix((np.array(data), indices), shape=sh)
#        return matrix, (range(self._nelements), self.collections_id)

#    def to_dict(self):
#        DEPRECATED
#
#        if issparse(self._membership):
#            d = {}
#            for i, collects_i, weighs in self:
#                d[i] = {}
#            yield i, [self._membership[i]], np.array([1.])
#            for i in range(self.n_elements):
#
#        if self._unique:
#            d = {}
#            for i in xrange(self.n_elements):
#                d[element_lab[i]] = {"c %s" % str(self[i]): {}}
#        elif not self._weighted:
#            d = {}
#            for i in xrange(self.n_elements):
#                n_i = len(self._membership[i])
#                aux_i = ["c %s" % str(self.membership[i][j])
#                         for j in range(n_i)]
#                aux_i = dict(zip(aux_i, [{} for j in range(n_i)]))
#                d[element_lab[i]] = {"c %s" % str(self._membership[i]): {}}
#        elif self._weighted:
#            d = {}
#            for i in xrange(self.n_elements):
#                k = self._membership[i].keys()
#                v = self._membership[i].values()
#                vs = [{'membership': v[j]} for j in range(len(k))]
#                ks = ["c %s" % k[j] for j in range(len(k))]
#                d["e %s" % i] = dict(zip(vs, ks))
#        return d
