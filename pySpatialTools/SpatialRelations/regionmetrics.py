
"""
RegionMetrics
-------------
The definition of distances between regions and the store of this measures
into an object.

TODO: Probably check indices-elements retrieving

"""

import networkx as nx
import numpy as np
from scipy.sparse import issparse


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
                 input_='indices', output='indices', _data=None, data_in=None):
        ## Relations management
        self.relations = relations
        if relations is not None:
            # Store associated data
            if issparse(relations):
                sh1 = relations.shape[1]
                if _data is None:
                    self._data = np.arange(sh1).reshape((sh1, 1))
                else:
                    if type(_data) == list:
                        _data = np.array(_data)
                    if len(_data.shape) == 1:
                        self._data = _data.reshape((len(_data), 1))
                    elif len(_data.shape) == 2:
                        self._data = _data
                    elif len(_data.shape) not in [1, 2]:
                        raise TypeError("Not correct shape of data.")
            else:
                if _data is None:
                    _data = np.arange(relations.shape[1])
                    self._data = _data.reshape((relations.shape[1], 1))
                else:
                    if type(_data) == list:
                        _data = np.array(_data)
                    if len(_data.shape) == 1:
                        self._data = _data.reshape((len(_data), 1))
                    elif len(_data.shape) == 2:
                        self._data = _data
                    elif len(_data.shape) not in [1, 2]:
                        raise TypeError("Not correct shape of data.")
            # Type of input
            if type(relations) == np.ndarray:
                self._store = 'matrix'
            elif type(relations) == nx.Graph:
                self._store = 'network'
            elif issparse(relations):
                self._store = 'sparse'
        if self._store != 'network':
            if data_in is None:
                data_in = np.arange(relations.shape[0])
                self._data_input = data_in.reshape((relations.shape[0], 1))
            else:
                if type(data_in) == list:
                    data_in = np.array(data_in)
                if len(data_in.shape) == 1:
                    self._data_input = data_in.reshape((len(data_in), 1))
                elif len(data_in.shape) == 2:
                    self._data_input = data_in
                elif len(data_in.shape) not in [1, 2]:
                    raise TypeError("Not correct shape of data input.")

        ## Type of values
        self._distanceorweighs = distanceorweighs
        if not distanceorweighs:
            self._null_value = 0.
            self._inv_null_value = np.inf
        self._symmetric = symmetric
        ## IO parameters
        self._input = input_
        self._out = output

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

    ############################# Transformation ##############################
    ###########################################################################
    def transform(self, f_trans, params={}):
        ##TODO:
        return f_trans(self.relations)

    ############################ Class properties #############################
    ###########################################################################
    @property
    def data(self):
        return self._data

    @property
    def data_input(self):
        if self._data_input is None:
            return self._data
        else:
            return self._data_input

    @property
    def data_output(self):
        return self._data

    @property
    def shape(self):
        return (len(self.data_input), len(self.data_output))


class DummyRegDistance(RegionDistances):
    """Dummy abstract region distance."""

    def __init__(self, regs):
        if type(regs) not in [np.ndarray, list]:
            raise TypeError("Incorrect ids of elements.")
        if type(regs) == list:
            regs = np.array(regs)
        if len(regs.shape) not in [1, 2]:
            raise TypeError("Incorrect shape of elements id.")
        if len(regs.shape) == 1:
            regs = regs.reshape((len(regs), 1))
        self._data = regs

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
        neighs = [reg]
        dists = [1.]
        neighs, dists = np.array(neighs).ravel(), np.array(dists)
        return neighs, dists
