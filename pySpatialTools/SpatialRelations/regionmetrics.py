
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

    def __init__(self, relations, distanceorweighs=True, symmetric=True,
                 input_='indices', output='indices', _data=None, data_in=None,
                 input_type=None, store=None):
        ## Initialization
        self._initialization()
        ## Relations management
        self.relations = relations
        self._format_relations(relations, _data)
        self._format_data_input(relations, data_in)
        self._format_retrieve_interactors()
        self._format_retrieve_filters(input_type, input_)
        ## Type of values
        self.set_distanceorweighs(distanceorweighs)
        self._symmetric = symmetric
        ## IO parameters
        self._input = input_
        self._out = output

    def _initialization(self):
        self._store = 'matrix'  # sparse, network
        self.relations = None
        self._data = None
        self._data_input = None
        self._out = 'indices'  # indices, elements_id
        self._input = 'indices'  # indices, elements_id

    def set_inout(self, input_type=None, input_=None, output=None):
        if input_ is not None:
            self._format_retrieve_filters(input_type, input_)
        if output is not None:
            self._out = output

    def set_distanceorweighs(self, distanceorweighs):
        """Setting edges type. Edges can represent similarity or distances."""
        self._distanceorweighs = distanceorweighs
        if distanceorweighs:
            self._distanceorweighs = True
            self._null_value = 0.
            self._inv_null_value = np.inf
        else:
            self._distanceorweighs = False
            self._null_value = np.inf
            self._inv_null_value = 0.

    ################################ Formatters ###############################
    ###########################################################################
    def _format_relations(self, relations, _data):
        """Format main stored object relations."""
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
                    if type(relations) == np.ndarray:
                        _data = np.arange(relations.shape[1])
                        self._data = _data.reshape((relations.shape[1], 1))
                    elif type(relations) == nx.Graph:
                        _data = np.array(relations.nodes())
                        self._data = _data.reshape((len(_data), 1))
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

    def _format_data_input(self, relations, data_in):
        """Format data input."""
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

    def _format_retrieve_interactors(self):
        """Format main functions of retrieving."""
        ## Format interactor functions with stored data
        if self._store == 'matrix':
            self.retrieve_neighs_spec = self._matrix_retrieve_neighs
        elif self._store == 'network':
            self.retrieve_neighs_spec = self._netx_retrieve_neighs
        elif self._store == 'sparse':
            self.retrieve_neighs_spec = self._sparse_retrieve_neighs

    def _format_retrieve_filters(self, input_type, input_):
        """Format main functions of inputs."""
        ## Format inputters
        if input_type is None:
            if input_ == 'indices':
                self.filter_reg = self._general_filter_indices_reg
                self._slice_transform = self._slice_transform_list
            elif input_ == 'elements_id':
                self.filter_reg = self._general_filter_elements_reg
                self._slice_transform = self._slice_transform_array
            else:
                self.filter_reg = self._general_filter_reg
                self._slice_transform = self._slice_transform_list
        elif input_type == 'general':
            self.filter_reg = self._general_filter_reg
            self._slice_transform = self._slice_transform_list
        elif input_type in ['int', 'integer']:
            self.filter_reg = self._int_filter_reg
            self._slice_transform = self._slice_transform_list
        elif input_type == 'array':
            self.filter_reg = self._array_general_filter_reg
            self._slice_transform = self._slice_transform_array
        elif input_type == 'array1':
            self.filter_reg = self._array1_filter_reg
            self._slice_transform = self._slice_transform_array
        elif input_type == 'array2':
            self.filter_reg = self._array2_filter_reg
            self._slice_transform = self._slice_transform_array
        elif input_type == 'list':
            self.filter_reg = self._list_filter_reg
            self._slice_transform = self._slice_transform_list
        elif input_type == 'list_int':
            self.filter_reg = self._list_int_filter_reg
            self._slice_transform = self._slice_transform_list
        elif input_type == 'list_array':
            self.filter_reg = self._list_array_filter_reg
            self._slice_transform = self._slice_transform_array

    ################################## Getters ################################
    ###########################################################################
    def __getitem__(self, i):
        if type(i) in [list, np.ndarray]:
            neighs, dists = self.retrieve_neighs(i)
        elif type(i) == int:
            if self.shape[0] <= i or i < 0:
                raise IndexError('Index i out of bounds.')
            neighs, dists = self.retrieve_neighs(i)
        elif isinstance(i, slice):
            start, stop, step = i.start, i.stop, i.step
            step = 1 if step is None else step
            idxs = self._slice_transform(start, stop, step)
            neighs, dists = self[idxs]
        else:
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
        reg = self.filter_reg(reg)
        ## 1. Perform the retrieve
        neighs, dists = self.retrieve_neighs_spec(reg)
#        print dists, reg
#        assert(all([len(e.shape) == 2 for e in dists]))
#        assert(all([len(e) == 0 for e in dists if np.prod(e.shape) == 0]))
        return neighs, dists

    ######################## Input filtering candidates #######################
    ###########################################################################
    ## Specific for each possible input for retrieve
    def _general_filter_reg(self, reg):
        ## 0. Format input
        if type(reg) == list:
            reg = self._list_filter_reg(reg)
        elif type(reg) == np.ndarray:
            reg = self._array_general_filter_reg(reg)
        elif type(reg) == int:
            reg = self._int_filter_reg(reg)
        return reg

    def _general_filter_indices_reg(self, reg):
        if type(reg) == list:
            reg = self._list_filter_reg(reg)
        elif type(reg) == int:
            reg = self._int_filter_reg(reg)
        return reg

    def _general_filter_elements_reg(self, reg):
        if type(reg) in [int, np.int32, np.int64]:
            reg = self._int_filter_reg(reg)
        else:
            reg = self._array_general_filter_reg(reg)
        return reg

    def _int_filter_reg(self, reg):
        reg = self._data[reg].ravel()
        return reg

    def _array2_filter_reg(self, reg):
        return reg.astype(int).ravel()

    def _array1_filter_reg(self, reg):
        return reg.astype(int)

    def _array_general_filter_reg(self, reg):
        if len(reg.shape) == 1 and reg.shape[0] == 1:
            reg = self._array1_filter_reg(reg)
        else:
            reg = self._array2_filter_reg(reg)
        return reg

    def _list_filter_reg(self, reg):
        if all([type(r) in [int, np.int32, np.int64] for r in reg]):
            reg = self._list_int_filter_reg(reg)
        elif all([type(r) == np.ndarray for r in reg]):
            reg = self._list_array_filter_reg(reg)
        else:
            raise TypeError("Incorrect input.")
        return reg

    def _list_array_filter_reg(self, reg):
        new_reg = []
        for i in range(len(reg)):
            new_reg.append(self._array_general_filter_reg(reg[i]))
        return new_reg

    def _list_int_filter_reg(self, reg):
        new_reg = []
        for i in range(len(reg)):
            new_reg.append(self._int_filter_reg(reg[i]))
        return new_reg

    ################ Slice formatting
    def _slice_transform_list(self, start, stop, step):
        return list(range(start, stop, step))

    def _slice_transform_array(self, start, stop, step):
        return np.array(range(start, stop, step))

    ##################### Interactors relations candidates ####################
    ###########################################################################
    ## Specific for each possible stored type of data
    ## Output neighs, dists: list of arrays [iss_i][neighs_i]
    def _matrix_retrieve_neighs(self, regs):
        neighs, dists = [], []
        for reg in regs:
            ## Check if it is in the data
            if reg not in self._data[:, 0]:
                if self._out == 'indices':
                    neighs.append(np.array([]).astype(int))
                else:
                    neighs.append(np.array([]))
                dists.append(np.array([[]]).T)
                continue
            logi = (self._data == reg).ravel()
            logi = self.relations[logi] != self._null_value
            logi = logi[:, 0]
            ## Formatting properly neighs
            if self._out == 'elements_id':
                neighs_r = self._data[logi, 0]
            else:
                neighs_r = np.where(logi)[0].astype(int)
            dists_r = self.relations[self._data[:, 0] == reg, logi]
            ## Storing final result
            neighs.append(np.array(neighs_r))
            dists.append(np.array([dists_r]).T)
#        #print dists, regs, logi
#        assert(all([len(e.shape) == 2 for e in dists]))
#        assert(all([len(e) == 0 for e in dists if np.prod(e.shape) == 0]))
#        if self._out == 'indices':
#            print neighs
#            inttypes = [int, np.int32, np.int64]
#            correcness = []
#            for nei in neighs:
#                if len(nei):
#                    correcness.append(all([type(e) in inttypes for e in nei]))
#                else:
#                    correcness.append(nei.dtype in inttypes)
#            assert(correcness)
        return neighs, dists

    def _sparse_retrieve_neighs(self, regs):
        neighs, dists = [], []
        for reg in regs:
            ## Check if it is in the data
            if reg not in self._data[:, 0]:
                if self._out == 'indices':
                    neighs.append(np.array([]).astype(int))
                else:
                    neighs.append(np.array([]))
                dists.append(np.array([[]]).T)
                continue
            i_reg = np.where(self._data[:, 0] == reg)[0][0]
            idxs = self.relations.getrow(i_reg).nonzero()[1].astype(int)
            if self._symmetric:
                idxs2 = self.relations.getcol(i_reg).nonzero()[0]
                idxs = np.unique(np.hstack([idxs, idxs2]))
            dists_r = [self.relations.getrow(i_reg).getcol(i).A[0, 0]
                       for i in idxs]
            ## Formatting properly neighs
            if self._out == 'elements_id':
                neighs_r = self._data[idxs, 0]
            else:
                neighs_r = idxs.astype(int)
            ## Storing final result
            neighs.append(np.array(neighs_r))
            dists.append(np.array([dists_r]).T)
#        assert(all([len(e.shape) == 2 for e in dists]))
#        assert(all([len(e) == 0 for e in dists if np.prod(e.shape) == 0]))
#        if self._out == 'indices':
#            print neighs, regs
#            inttypes = [int, np.int32, np.int64]
#            correcness = []
#            for nei in neighs:
#                if len(nei):
#                    correcness.append(all([type(e) in inttypes for e in nei]))
#                else:
#                    correcness.append(nei.dtype in inttypes)
#            print correcness, neighs, regs, len(neighs), len(regs)
#            #assert(correcness)
#            assert(len(neighs) == len(regs))
        return neighs, dists

    def _netx_retrieve_neighs(self, regs):
        neighs, dists = [], []
        for reg in regs:
            ## Check if it is in the data
            if reg not in self._data[:, 0]:
                if self._out == 'indices':
                    neighs.append(np.array([]).astype(int))
                else:
                    neighs.append(np.array([]))
                dists.append(np.array([[]]).T)
                continue
            reg = int(reg) if type(reg) == np.ndarray else reg
            neighs_r = self.relations.neighbors(reg)
            dists_r = [self.relations[reg][nei]['weight']
                       for nei in neighs_r]
            ## Formatting properly neighs
            if self._out == 'indices':
                neighs_aux = []
                for nei in neighs_r:
                    neighs_aux.append(np.where(nei == self.data)[0])
                neighs_r = np.concatenate(neighs_aux).astype(int)
            ## Storing final result
            neighs.append(np.array(neighs_r))
            dists.append(np.array([dists_r]).T)
#        print dists
#        assert(all([len(e.shape) == 2 for e in dists]))
#        assert(all([len(e) == 0 for e in dists if np.prod(e.shape) == 0]))
#        if self._out == 'indices':
#            print neighs
#            inttypes = [int, np.int32, np.int64]
#            correcness = []
#            for nei in neighs:
#                if len(nei):
#                    correcness.append(all([type(e) in inttypes for e in nei]))
#                else:
#                    correcness.append(nei.dtype in inttypes)
#            assert(correcness)
        return neighs, dists

#    def _general_retrieve_neighs(self, reg):
#        if self._store == 'matrix':
#            neighs, dists = self._matrix_retrieve_neighs(reg)
#        elif self._store == 'sparse':
#            neighs, dists = self._sparse_retrieve_neighs(reg)
#        elif self._store == 'network':
#            neighs, dists = self._netx_retrieve_neighs(reg)
#        return neighs, dists

    ############################# Transformation ##############################
    ###########################################################################
    def transform(self, f_trans, params={}):
        """Application of a transformation to the relations."""
        return f_trans(self.relations, *params)

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

    def __init__(self, regs, input_type=None):
        self._initialization()
        if type(regs) not in [np.ndarray, list]:
            raise TypeError("Incorrect ids of elements.")
        if type(regs) == list:
            regs = np.array(regs)
        if len(regs.shape) not in [1, 2]:
            raise TypeError("Incorrect shape of elements id.")
        if len(regs.shape) == 1:
            regs = regs.reshape((len(regs), 1))
        self._data = regs
        self._format_retrieve_filters(input_type, input_=None)

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
        reg = self.filter_reg(reg)
        ## 1. Perform the retrieve
        neighs = list(np.array(reg).reshape((len(reg), 1)))
        dists = [np.array([[1.]*len(neighs)]).T]*len(reg)
        return neighs, dists
