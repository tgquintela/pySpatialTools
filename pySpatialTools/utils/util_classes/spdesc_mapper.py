
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

inttypes = [int, np.int32, np.int64]


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
        elif type(keys) in [list, tuple]:
            assert(all([type(k) == int for k in keys]))
            outs = [self._mapper(k) for k in keys]
        else:
            raise TypeError("Not correct input for spatial descriptor mapper.")
        return outs

    def set_from_array(self, array_mapper, _n_vars_out):
        """Set mapper from array."""
        assert(len(array_mapper.shape) == 2)
        if array_mapper.shape[1] != _n_vars_out:
            msg = "Not correct shape of array to be a selector mapper."
            raise TypeError(msg)
        array_mapper = array_mapper.astype(int)
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
        res = tuple(res)
        return res

    def _formatting_unique_collective_mapper(self, mapper):
        """Is a collection of mappers categorize by a only mapper."""
        self.assert_correctness = self.assert_correctness_mapper
        if type(mapper).__name__ == 'instance':
            self._mapper_setting(mapper)
            self.__getitem__ = self._getitem_mapper
        elif type(mapper) == tuple:
            if type(mapper[0]) == int:
                assert(len(mapper) == sum(self._n_vars_out))
                self._mapper_setting(mapper)
                self.__getitem__ = self._getitem_mapper
            else:
                assert(type(mapper[1]) == dict)
                assert(type(mapper[0]).__name__ == 'function')
                self._mapper_setting(mapper[0])
                self._initialize_variables(**mapper[1])
                self.__getitem__ = self._getitem_mapper
        elif type(mapper) == np.ndarray:
            self._mapper_setting(mapper)
            self.__getitem__ = self._getitem_mapper
        elif type(mapper).__name__ == 'function':
            self._mapper_setting(mapper)
            self.__getitem__ = self._getitem_mapper

    def assert_correctness(self, object2manageselection):
        for i in range(len(self.selectors)):
            self.selectors[i].assert_correctness(object2manageselection)

    def assert_correctness_mapper(self, obj):
        pass

    def _preprocess_selector(self, map_sel, map_sel_type):
        if map_sel is None:
            map_sel = map_sel_type(map_sel)
        elif isinstance(map_sel, map_sel_type):
            pass
        elif type(map_sel) == list:
            map_sel = map_sel_type(*map_sel)
        else:
            map_sel = map_sel_type(map_sel)
        return map_sel

    def _getitem_mapper(self, keys):
        if type(keys) == int:
            outs = self._mapper(keys)
            outs = self._outformat_mapper(outs)
        elif type(keys) in [list, tuple]:
            assert(all([type(k) == int for k in keys]))
            outs = []
            for k in keys:
                outs.append(self._outformat_mapper(self._mapper(k)))
        else:
            msg = "Not correct input for %s selector." % self.__name__
            raise TypeError(msg)
        return outs

    def _outformat_mapper(self, out):
        out_format = _outformat_mapper(out, self._n_vars_out)
#        if type(out) == tuple:
#            assert(len(out) == sum(self._n_vars_out))
#            out_format = []
#            limits = [0] + list(np.cumsum(self._n_vars_out))
#            for i in range(len(self._n_vars_out)):
#                aux_out = out[limits[i]:limits[i+1]]
#                out_format.append(aux_out)
#            out_format = tuple(out_format)
#        else:
#            assert(type(out) == list)
#            assert(len(out) == len(self._n_vars_out))
#            assert(all([len(out[i]) == self._n_vars_out[i]
#                        for i in range(len(out))]))
#            out_format = tuple(out)
        return out_format

    def _initialize_variables(self, n_in=None, n_out=None):
        if n_in is not None:
            self.n_in = n_in
#            if '_array_mapper' in dir(self):
#                if self._array_mapper is not None:
#                    assert(len(self._array_mapper) >= n_in)

    def _mapper_setting(self, mapper):
        if type(mapper) == np.ndarray:
            self._array_mapper = mapper
            self._mapper =\
                lambda idx: tuple(self._array_mapper[idx].astype(int))
        elif type(mapper).__name__ == 'function':
            self._mapper = mapper
        elif type(mapper) == tuple:
            self._mapper = lambda idx: mapper
        else:
            assert(type(mapper).__name__ == 'instance')
            self._mapper = mapper._mapper
#            if '_array_mapper' in dir(mapper):
#                if mapper._array_mapper is not None:
#                    self._array_mapper = mapper._array_mapper
#                    self._mapper = lambda idx: tuple(self._array_mapper[idx])
            self.n_in = mapper.n_in


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
        assert(len(manager.retrievers) >= self.n_out[0])
        if '_array_mapper' in dir(self):
            if self._array_mapper is not None:
                for i in self._pos_out[0]:
                    idxs = np.where(self._array_mapper[:, 0] == i)
                    max_out = np.max(self._array_mapper[idxs, 1])
                    assert(len(manager.retrievers[i]._output_map) > max_out)


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
        assert(len(manager._maps_input) >= self.n_out[0])
        assert(len(manager.features) >= self.n_out[1])


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
        assert(2 >= self.n_out[0])  # It is a boolean variable
        assert(len(manager.features) >= self.n_out[1])


class Feat_RetrieverSelector(GeneralCollectionSelectors):
    """Features retriever mapper to indicate the path of possible options to
    interact with features.
    """
    _mapper = lambda s, idx: [(0, 0)]*3
    __name__ = "pst.Feat_RetrieverSelector"

    def _inititizalization(self):
        self._default_map_values = [(0, 0)]*3
        self._n_vars_out = [2, 2, 2]
        self.n_out = []
        self._pos_out = []
        self.n_in = 0
        self._open_n = False

    def __init__(self, mapper_featin, mapper_featout=None, mapper_desc=None):
        ## Initialization
        self._inititizalization()
#        print '-'*50
#        print mapper_featin, mapper_featout, mapper_desc
#        if type(mapper_featin) == np.ndarray and mapper_featout is None:
#            raise Exception("Quieto parau")
        if mapper_featout is None:
            self._formatting_unique_collective_mapper(mapper_featin)
        else:
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


class Sp_DescriptorSelector(GeneralCollectionSelectors):
    """Spatial descriptor mapper to indicate the path of possible options to
    compute spatial descriptors.
    """
    _mapper = lambda s, idx: (0, 0, 0, 0, 0, 0, 0, 0)
    __name__ = "pst.Sp_DescriptorSelector"

    def _initialization(self):
        self._n_vars_out = [2, [2, 2, 2]]
        self._array_mapper = None
        self.n_in = 0

    def __init__(self, map_ret=None, map_feat=None):
        self._initialization()
#        print '.'*50
#        print map_ret, map_feat
#        print '.'*50
        if map_feat is None:
            self._formatting_unique_collective_mapper(map_ret)
        else:
            map_ret =\
                self._preprocess_selector(map_ret, Spatial_RetrieverSelector)
            map_feat =\
                self._preprocess_selector(map_feat, Feat_RetrieverSelector)
            self.selectors = map_ret, map_feat


def _outformat_mapper(out, _n_vars_out):
    """Outformat mapper."""
    if type(out) == list:
        outs = []
        for i in range(len(out)):
            outs.append(_outformat_mapper(out[i], _n_vars_out))
    elif type(out) == tuple:
        if all([type(o) in inttypes for o in out]):
            init, outs = 0, []
            for i in range(len(_n_vars_out)):
                if type(_n_vars_out[i]) in inttypes:
                    aux_out = out[init:(init+_n_vars_out[i])]
                    outs.append(aux_out)
                    init += _n_vars_out[i]
                else:
                    aux_out = out[init:(init+sum(_n_vars_out[i]))]
                    outs.append(_outformat_mapper(aux_out, _n_vars_out[i]))
                    init += sum(_n_vars_out[i])
            outs = tuple(outs)
        else:
            assert(len(_n_vars_out) == len(out))
            outs = out
    return outs
