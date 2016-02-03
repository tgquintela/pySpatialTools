
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
from scipy.sparse import coo_matrix


class Membership:
    """Class representing a membership object in which maps every element
    assigned by an index to a group or collection of elements.
    Represent a mapping between elements and collections.

    """

    _membership = []
    _nelements = 0
    _unique = True
    _weighted = False
    _ncollections = 0
    _maxcollection_id = -1

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
        if type(membership) == np.ndarray:
            self._membership = membership
            self._nelements = membership.shape[0]
            self._unique = True
            self._weighted = False
            self._ncollections = np.unique(membership).shape[0]
            self._maxcollection_id = np.max(membership)
        elif type(membership) == list:
            self._nelements = len(membership)
            types = np.array([type(e) for e in membership])
            if np.any(types == list):
                for i in xrange(len(membership)):
                    if type(membership[i]) != list:
                        membership[i] = [membership[i]]
            if np.all(types == dict):
                self._membership = membership
                self._unique = False
                self._weighted = True
                self._nelements = len(membership)
                aux = [membership[i].keys() for i in xrange(self._nelements)]
                aux = np.hstack(aux)
                self._ncollections = np.unique(aux).shape[0]
                self._maxcollection_id = np.max(aux)
            if np.all(types == list):
                length = np.array([len(e) for e in membership])
                if np.all(length == 1):
                    membership = np.array(membership)
                    self._membership = membership
                    self._nelements = membership.shape[0]
                    self._unique = True
                    self._weighted = False
                    self._ncollections = np.unique(membership).shape[0]
                    self._maxcollection_id = np.max(membership)
                else:
                    self._membership = membership
                    self._nelements = len(membership)
                    self._unique = False
                    self._weighted = False
                    aux = np.hstack(membership)
                    self._ncollections = np.unique(aux).shape[0]
                    maxs_id = [np.max(e) for e in membership if e]
                    self._maxcollection_id = np.max(maxs_id)

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
        if i < 0 or i >= self._nelements:
            raise IndexError("Element ID out of range")
        if self._unique:
            collections = [self._membership[i]]
        else:
            if not self._weighted:
                collections = self._membership[i]
            else:
                collections = self._membership[i].keys()
        return collections

    def __eq__(self, collection_id):
        elements = self.getcollection(collection_id)
        return elements

    def __iter__(self):
        """Iterates over the elements and return element i, its collections to
        which it belongs and the membership value.
        """
        if self._unique:
            for i in xrange(self._nelements):
                yield i, [self._membership[i]], np.array([1.])
        else:
            if not self._weighted:
                for i in xrange(self.n_elements):
                    colls = self._membership[i]
                    memb_vals = np.ones(len(colls))
                    yield i, colls, memb_vals
            else:
                for i in xrange(self._nelements):
                    colls = self._membership[i].keys()
                    memb_vals = np.array(self._membership[i].values())
                    yield i, colls, memb_vals

    def __str__(self):
        "Return a representation of information about the data of this class."
        summary = """Membership class with %s elements and %s collections,
            in which there are %sweighted and %sunique relations between
            elements and collections."""
        wei = "" if self._weighted else "non-"
        unique = "" if self._unique else "non-"
        summary = summary % (self._nelements, self._ncollections, wei, unique)
        return ""

    @property
    def membership(self):
        """Returns the membership data."""
        return self._membership

    @property
    def n_elements(self):
        """Returns the number of elements."""
        return self._nelements

    @property
    def n_collections(self):
        "Returns the number of collections."
        return self._ncollections

    @property
    def collections_id(self):
        "Return the collections ID of the membership."
        # For each 3 cases
        if self._unique:
            collections_id = np.unique(self._membership)
        elif not self._unique and not self._weighted:
            collections_id = np.unique(np.hstack(self._membership)).astype(int)
        elif not self._unique and self._weighted:
            aux = [self._membership[e].keys() for e in self._membership]
            collections_id = np.unique(np.hstack(aux))
        return collections_id

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
        if collection_id < 0 or collection_id > self._maxcollection_id:
            raise IndexError("Collection ID out of range")
        if self._unique:
            elements = self._membership == collection_id
        else:
            if not self._weighted:
                elements = [collection_id in self._membership[i]
                            for i in xrange(self._nelements)]
                elements = np.array(elements).astype(bool)
            else:
                aux = [self._membership[e].keys() for e in self._membership]
                elements = [collection_id in aux[i]
                            for i in xrange(self._nelements)]
                elements = np.array(elements).astype(bool)
        return elements

    def to_network(self):
        "Return a networkx graph object."
        G = nx.from_dict_of_dicts(self.to_dict)
        return G

    def to_dict(self):
        "Return a dictionary object."
        element_lab = ["e %s" % str(e) for e in xrange(self._nelements)]
        if self._unique:
            d = {}
            for i in xrange(self._nelements):
                d[element_lab[i]] = {"c %s" % str(self._membership[i]): {}}
        elif not self._weighted:
            d = {}
            for i in xrange(self._nelements):
                n_i = len(self._membership[i])
                aux_i = ["c %s" % str(self.membership[i][j])
                         for j in range(n_i)]
                aux_i = dict(zip(aux_i, [{} for j in range(n_i)]))
            d[element_lab[i]] = {"c %s" % str(self._membership[i]): {}}
        elif self._weighted:
            d = {}
            for i in xrange(self._nelements):
                k = self._membership[i].keys()
                v = self._membership[i].values()
                vs = [{'membership': v[j]} for j in range(len(k))]
                ks = ["c %s" % k[j] for j in range(len(k))]
                d["e %s" % i] = dict(zip(vs, ks))
        return d

    def to_sparse(self):
        "Return a sparse matrix object."
        sh = self._nelements, self._ncollections
        aux_map = dict(zip(self.collections_id, range(self._ncollections)))
        if self._unique:
            matrix = np.array([aux_map[e] for e in self._membership])
            matrix = matrix.astype(int)
            matrix = coo_matrix((np.ones(sh[0]), (range(sh[0], matrix))),
                                shape=sh)
        elif not self._weighted:
            indices = []
            for i in xrange(sh[0]):
                for j in range(self._membership[i]):
                    indices.append((i, aux_map[self._membership[i][j]]))
            matrix = coo_matrix((np.ones(sh[0]), indices), shape=sh)

        elif self._weighted:
            indices, data = [], []
            for i in xrange(sh[0]):
                for j in self._membership[i]:
                    indices.append((i, aux_map[j]))
                    data.append(self._membership[i][j])
            matrix = coo_matrix((np.array(data), indices), shape=sh)
        return matrix, (range(self._nelements), self.collections_id)

    def reverse_mapping(self):
        """Reverse the mapping of elements to collections to collections to
        elements.
        """
        reversed_ = {}
        for coll_id in self.collections_id:
            elements = list(np.where(self.getcollection(coll_id))[0])
            reversed_[coll_id] = elements
        return reversed_


def formatrelationship(relationship, typeoutput):
    pass


"""
typeoutput = 'network', 'sparse', 'membership', 'list', 'matrix'
"""
