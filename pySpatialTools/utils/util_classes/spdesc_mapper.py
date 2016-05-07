
"""
spatial descriptors mapper
--------------------------
Module which contains auxiliar functions and classes to use different spatial
descriptors and models for the same data.
This mappers are used to select for each element the most alike method to use
using pre-stablished criteria described in this object.
It is essential to compute complex descriptors from different scale spatial
data.

"""

import numpy as np


#class Sp_DescriptorMapper:
#    """Spatial descriptor mapper to indicate the path of possible options to
#    compute spatial descriptors.
#    """
#    _mapper = lambda s, idx: (0, 0, 0, 0, 0, 0, 0, 0)
#    __name__ = "pst.Sp_DescriptorMapper"
#
#    def __init__(self, staticneighs=None, mapretinput=None, mapretout=None,
#                 mapfeatinput=None, mapfeatoutput=None):
#
#        dummymapper = lambda idx: 0
#
#        if staticneighs is None:
#            if type(staticneighs) == np.ndarray:
#                staticneighs = lambda idx: staticneighs[idx]
#            if type(staticneighs).__name__ == 'function':
#                pass
#            else:
#                staticneighs = dummymapper
#
#        if mapretinput is None:
#            if type(mapretinput) == np.ndarray:
#                mapretinput = lambda idx: mapretinput[idx]
#            if type(mapretinput).__name__ == 'function':
#                pass
#            else:
#                mapretinput = dummymapper
#
#        if mapretout is None:
#            if type(mapretout) == np.ndarray:
#                mapretout = lambda idx: mapretout[idx]
#            if type(mapretout).__name__ == 'function':
#                pass
#            else:
#                mapretout = dummymapper
#
#        if mapfeatinput is None:
#            if type(mapfeatinput) == np.ndarray:
#                mapfeatinput = lambda idx: mapfeatinput[idx]
#            if type(mapfeatinput).__name__ == 'function':
#                pass
#            else:
#                mapfeatinput = dummymapper
#
#        if mapfeatoutput is None:
#            if type(mapfeatoutput) == np.ndarray:
#                mapfeatoutput = lambda idx: mapfeatoutput[idx]
#            if type(mapfeatoutput).__name__ == 'function':
#                pass
#            else:
#                mapfeatoutput = dummymapper
#
#        self._mapper = lambda i: (staticneighs(i), mapretinput(i),
#                                  mapretout(i), mapfeatinput(i),
#                                  mapfeatoutput(i))
#
#    def __getitem__(self, keys):
#        if type(keys) == int:
#            istatic, iret, irout, ifeat, ifout = self._mapper(keys)
#        else:
#            raise TypeError("Not correct input for spatial descriptor mapper.")
#        return istatic, iret, irout, ifeat, ifout
#
#    def set_from_array(self, array_mapper):
#        "Set mapper from array."
#        if array_mapper.shape[1] != 5:
#            msg = "Not correct shape of array to be a spatial mapper."
#            raise TypeError(msg)
#        self._mapper = lambda idx: tuple(array_mapper[idx])
#
#    def set_from_function(self, function_mapper):
#        try:
#            a, b, c, d, e = function_mapper(0)
#            self._mapper = function_mapper
#        except:
#            raise TypeError("Incorrect function mapper.")
#
#    def checker(self, constraints):
#        "TODO: checker functions if this mapper selector fits the constraints."
#        pass
#
#    def set_default_with_constraints(self, constraints):
#        "TODO: default builder of the but with constraints."
#        pass


class GeneralSelector:
    """General selector."""

    def __init__(self, mapper, n_in=None, n_out=None, compute=False):
        ## Preparation
        self._inititizalization()
        ## Formatting and storing
        self._format_maps(mapper, n_in, n_out, compute)

    def _format_maps(self, mapper, n_in, n_out, compute):
        if type(mapper) == np.ndarray:
            if len(mapper.shape) == 1:
                mapper = mapper.reshape((len(mapper), 1))
        self._define_lack_parameters(mapper)
        ## Formatting and storing parameters
        if type(mapper) == np.ndarray:
            self.set_from_array(mapper, self._n_vars_out)
        elif type(mapper).__name__ == 'function':
            if n_out is None:
                n_out = tuple([None]*self._n_vars_out)
            elif type(n_out) == int:
                n_out = [n_out]
            assert(len(n_out) == self._n_vars_out)
            self.set_from_function(mapper, n_in, n_out, self._n_vars_out,
                                   compute)

    def __getitem__(self, keys):
        if type(keys) == int:
            outs = self._mapper(keys)
        else:
            raise TypeError("Not correct input for spatial descriptor mapper.")
        return outs

    def set_from_array(self, array_mapper, _n_vars_out):
        """Set mapper from array."""
        assert(len(array_mapper.shape) == 2)
        if array_mapper.shape[1] != _n_vars_out:
            msg = "Not correct shape of array to be a spatial mapper."
            raise TypeError(msg)
        self._array_mapper = array_mapper
        self._mapper = lambda idx: tuple(self._array_mapper[idx])
        self._pos_out = []
        for i in range(_n_vars_out):
            self._pos_out.append(np.unique(array_mapper[:, i]))
        self.close_output = True
        self.n_in = len(array_mapper)
        self.n_out = [len(self._pos_out[i]) for i in range(len(self._pos_out))]

    def set_from_function(self, mapper, n_in, n_out, _n_vars_out,
                          compute=True):
        """Set mapper from function."""
        assert(len(n_out) == _n_vars_out)
        self._mapper = mapper
        self.n_in = n_in
        out_uniques = [[]]*_n_vars_out
        if compute:
            self.close_output = True
            if type(n_in) == int:
                for i in range(n_in):
                    out = self._mapper[i]
                    for j in range(len(out)):
                        if out[j] not in out_uniques[j]:
                            out_uniques[j].append(out[j])
            self._pos_out = out_uniques
            self.n_out = [len(e) for e in self._pos_out]

    def _define_lack_parameters(self, mapper):
        if '' not in dir(self):
            if type(mapper) == np.ndarray:
                out = mapper[0]
            else:
                out = mapper(0)
            out = np.array([out]).ravel()
            self._n_vars_out = len(out)


class GeneralCollectionSelectors:
    """General collections of selectors."""

    def __init__(self, selectors):
        n_ins = [selectors[i].n_in for i in range(len(selectors))]
        assert(all([n_in == n_ins[0] for n_in in n_ins]))
        self.n_in = n_ins[0]
        self.n_out, self._pos_out = [], []
        for i in range(len(selectors)):
            self.n_out += selectors[i].n_out
            self._pos_out += selectors[i]._pos_out
        self.selectors = selectors

    def assert_correctness(self, object2manageselection):
        for i in range(len(self.selectors)):
            self.selectors[i].assert_correctness(object2manageselection)


class DummySelector(GeneralSelector):
    """Dummy selector for testing and example purposes."""
    def _inititizalization(self):
        self.n_in = 0
        self.n_out = [1]
        self._pos_out = [[0]]
        self._open_n = (False)

    def set_pars(self, _n_vars_out, _default_map_values, n_out=None):
        self._default_map_values = _default_map_values
        self._n_vars_out = _n_vars_out
        if n_out is not None:
            n_out = [n_out] if type(n_out) == int else n_out
            assert(len(n_out) == self._n_vars_out)
            self._open_n = [range(n_out[i]) for i in range(self._n_vars_out)]
        self(self._mapper, n_out=n_out)


class Spatial_RetrieverSelector(GeneralSelector):
    """Spatial retriever mapper to indicate the path of possible options to
    retrieve spatial neighbourhood.

    2-dim selector output:
        - Retriever selection
        - Outputmap selection
    """
    _mapper = lambda s, idx: (0, 0)
    __name__ = "pst.Sp_RetrieverMapper"

    def _inititizalization(self):
        self._default_map_values = (0, 0)
        self._n_vars_out = 2
        self.n_in = 0
        self.n_out = [1, 1]
        self._pos_out[[0], [0]]
        self._open_n = (False, False)

    def __init__(self, _mapper_ret, mapper_out=None, n_in=None, n_out=None):
        ## Filter mappings
        if mapper_out is not None:
            assert(type(_mapper_ret) == type(mapper_out))
            if type(mapper_out) == np.ndarray:
                mapper = np.hstack([_mapper_ret, mapper_out]).T
            else:
                mapper = lambda idx: (_mapper_ret[idx], mapper_out[idx])
        else:
            mapper = _mapper_ret
        ## Creation of the mapper
        self._format_maps(mapper, n_in, n_out, compute=False)

    def assert_correctness(self, manager):
        assert(len(manager.retrievers) == self.n_out[0])
        if 'array_mapper' in dir(self):
            if self.array_mapper is not None:
                for i in self._pos_out[0]:
                    [np.where(self.array_mapper[:, 0] == i)]


class Feat_RetrieverSelector(GeneralSelector):
    """Features retriever mapper to indicate the path of possible options to
    interact with features.
    """
    _mapper = lambda s, idx: (0, 0)
    __name__ = "pst.Feat_RetrieverMapper"

    def _inititizalization(self):
        self._default_map_values = (0, 0)
        self._n_vars_out = 2
        self.n_out = [1, 1]
        self._pos_out[[0], [0]]
        self.n_in = 0
        self._open_n = (False, False)

    def __init__(self, _mapper_ret, mapper_out=None):
        ## Filter mappings
        if mapper_out is None:
            pass

    def assert_correctness(self, manager):
        assert(len(manager.retrievers) == self.n_out[0])

#        feat_manager
#        t_feat_in, t_feat_out, t_feat_des
#
#        for i
#        if self.close_output:
#            len(feat_manager._maps_input) == 
#            len(feat_manager.features)
#
#            len(feat_manager._maps_input)
#            len(feat_manager.features)
#
#            len(feat_manager.features)
#            len(feat_manager.descriptormodels)
