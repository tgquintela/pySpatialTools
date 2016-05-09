
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
from copy import copy


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
        if type(mapper) == tuple:
            assert(len(mapper) == self._n_vars_out)
            mapper = lambda idx: mapper
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
        else:
            assert(self.__name__ == mapper.__name__)
            assert('_mapper' in dir(mapper))
            self._n_vars_out = mapper._n_vars_out
            if '_array_mapper' in dir(mapper):
                self._format_maps(mapper._array_mapper, mapper.n_in,
                                  mapper.n_out, compute)
            else:
                self._format_maps(mapper._mapper, mapper.n_in, mapper.n_out,
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
            msg = "Not correct shape of array to be a selector mapper."
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
                    out = self._mapper(i)
                    for j in range(len(out)):
                        if out[j] not in out_uniques[j]:
                            out_uniques[j].append(out[j])
            self._pos_out = out_uniques
            self.n_out = [len(e) for e in self._pos_out]
        else:
            self._pos_out = [[0] for i in range(_n_vars_out)]
            self.n_out = [len(e) for e in self._pos_out]

    def _define_lack_parameters(self, mapper):
        if '_n_vars_out' not in dir(self):
            if type(mapper).__name__ == 'function':
                out = mapper(0)
            else:
                out = mapper[0]
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

    def __getitem__(self, idx):
        res = []
        for i in range(len(self.selectors)):
            res.append(self.selectors[i][idx])
        return res

    def assert_correctness(self, object2manageselection):
        for i in range(len(self.selectors)):
            self.selectors[i].assert_correctness(object2manageselection)


class DummySelector(GeneralSelector):
    """Dummy selector for testing and example purposes."""
    __name__ = "pst.DummySelector"

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
            self._pos_out = [range(n_out[i]) for i in range(self._n_vars_out)]
        self.__init__(self._mapper, n_out=n_out)


class Spatial_RetrieverSelector(GeneralSelector):
    """Spatial retriever mapper to indicate the path of possible options to
    retrieve spatial neighbourhood.

    2-dim selector output:
        - Retriever selection
        - Outputmap selection
    """
#    _mapper = lambda s, idx: (0, 0)
    __name__ = "pst.Spatial_RetrieverSelector"

    def _inititizalization(self):
        self._default_map_values = (0, 0)
        self._n_vars_out = 2
        self.n_in = 0
        self.n_out = [1, 1]
        self._pos_out = [[0], [0]]
        self._open_n = (False, False)

    def __init__(self, _mapper_ret, mapper_out=None, n_in=None, n_out=None):
        ## Initialization
        self._inititizalization()
        ## Filter mappings
        mapper = self._preformat_maps(_mapper_ret, mapper_out)
        ## Creation of the mapper
        self._format_maps(mapper, n_in, n_out, compute=False)

    def _preformat_maps(self, _mapper_ret, mapper_out):
        """Preformat input maps."""
        if mapper_out is not None:
            assert(type(_mapper_ret) == type(mapper_out))
            if type(mapper_out) == np.ndarray:
                assert(len(_mapper_ret) == len(mapper_out))
                mapper = np.vstack([_mapper_ret, mapper_out]).T
            elif type(mapper_out) == int:
                mapper = (_mapper_ret, mapper_out)
            else:
                mapper = lambda idx: (_mapper_ret(idx), mapper_out(idx))
        else:
            mapper = _mapper_ret
        return mapper

    def assert_correctness(self, manager):
        assert(len(manager.retrievers) == self.n_out[0])
        if 'array_mapper' in dir(self):
            if self.array_mapper is not None:
                for i in self._pos_out[0]:
                    [np.where(self.array_mapper[:, 0] == i)]


class FeatInd_RetrieverSelector(GeneralSelector):
    """Computation of features of the element we want to study."""

#    _mapper = lambda self, idx: (0, 0)
    __name__ = "pst.FeatInd_RetrieverSelector"

    def _inititizalization(self):
        self._default_map_values = (0, 0)
        self._n_vars_out = 2
        self.n_out = [1, 1]
        self._pos_out = [[0], [0]]
        self.n_in = 0
        self._open_n = (False, False)

    def __init__(self, _mapper_feats, mapper_inp=None, n_in=None, n_out=None):
        ## Initialization
        self._inititizalization()
        ## Filter mappings
        mapper = self._preformat_maps(_mapper_feats, mapper_inp)
        ## Creation of the mapper
        self._format_maps(mapper, n_in, n_out, compute=False)

    def _preformat_maps(self, _mapper_feats, mapper_inp):
        """Preformat input maps."""
        if mapper_inp is not None:
            assert(type(_mapper_feats) == type(mapper_inp))
            if type(mapper_inp) == np.ndarray:
                assert(len(_mapper_feats) == len(mapper_inp))
                mapper = np.vstack([_mapper_feats, mapper_inp]).T
            elif type(mapper_inp) == int:
                mapper = (_mapper_feats, mapper_inp)
            else:
                mapper = lambda idx: (_mapper_feats(idx), mapper_inp(idx))
        else:
            mapper = _mapper_feats
        return mapper

    def assert_correctness(self, manager):
        assert(len(manager._maps_input) == self.n_out[0])
        assert(len(manager.features) == self.n_out[1])


class Desc_RetrieverSelector(GeneralSelector):
    """Selection of descriptor computation after pointfeatures and
    neighbourhood features."""

    __name__ = "pst.Desc_RetrieverSelector"

    def _inititizalization(self):
        self._default_map_values = (0, 0)
        self._n_vars_out = 2
        self.n_out = [1, 1]
        self._pos_out = [[0], [0]]
        self.n_in = 0
        self._open_n = (False, False)

    def __init__(self, _mapper_feats, mapper_inp=None, n_in=None, n_out=None):
        ## Initialization
        self._inititizalization()
        ## Filter mappings
        mapper = self._preformat_maps(_mapper_feats, mapper_inp)
        ## Creation of the mapper
        self._format_maps(mapper, n_in, n_out, compute=False)

    def _preformat_maps(self, _mapper_feats, mapper_inp):
        """Preformat input maps."""
        if mapper_inp is not None:
            assert(type(_mapper_feats) == type(mapper_inp))
            if type(mapper_inp) == np.ndarray:
                assert(len(mapper_inp) == len(_mapper_feats))
                mapper = np.vstack([_mapper_feats, mapper_inp]).T
            elif type(mapper_inp) == int:
                mapper = (_mapper_feats, mapper_inp)
            else:
                mapper = lambda idx: (_mapper_feats(idx), mapper_inp(idx))
        else:
            mapper = _mapper_feats
        return mapper

    def assert_correctness(self, manager):
        assert(len(manager._maps_input) == self.n_out[0])
        assert(len(manager.features) == self.n_out[1])


class Feat_RetrieverSelector(GeneralCollectionSelectors):
    """Features retriever mapper to indicate the path of possible options to
    interact with features.
    """
    _mapper = lambda s, idx: (0, 0)
    __name__ = "pst.Feat_RetrieverSelector"

    def _inititizalization(self):
        self._default_map_values = (0, 0)
        self._n_vars_out = 2
        self.n_out = []
        self._pos_out = []
        self.n_in = 0
        self._open_n = False

    def __init__(self, mapper_featin, mapper_featout, mapper_desc):
        ## Instantiation
        mapper_featin = FeatInd_RetrieverSelector(mapper_featin)
        mapper_featout = FeatInd_RetrieverSelector(mapper_featout)
        mapper_desc = Desc_RetrieverSelector(mapper_desc)
        ## Assert correct inputs
        self._assert_inputs(mapper_featin, mapper_featout, mapper_desc)
        ## Set informative parameters
        self._set_feat_selector(mapper_featin, mapper_featout, mapper_desc)

    def _assert_inputs(self, mapper_featin, mapper_featout, mapper_desc):
        assert(mapper_featin.__name__ == "pst.FeatInd_RetrieverSelector")
        assert(mapper_featout.__name__ == "pst.FeatInd_RetrieverSelector")
        assert(mapper_desc.__name__ == "pst.Desc_RetrieverSelector")
        if mapper_featin.n_in is not None and mapper_featin.n_in != 0:
            assert(type(mapper_featin.n_in) == int)
            assert(mapper_featin.n_in == mapper_featout.n_in)
            assert(mapper_featin.n_in == mapper_desc.n_in)

    def _set_feat_selector(self, mapper_featin, mapper_featout, mapper_desc):
        self.n_in = mapper_featin.n_in
        self.n_out = mapper_featin.n_out+mapper_featout.n_out+mapper_desc.n_out
        self._pos_out =\
            mapper_featin._pos_out+mapper_featout._pos_out+mapper_desc._pos_out
        self.selectors = mapper_featin, mapper_featout, mapper_desc
