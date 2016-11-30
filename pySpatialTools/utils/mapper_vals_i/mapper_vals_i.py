
"""
Mapper vals i
-------------
Module which contains mapper for vals i. This Mapper is important because is
the one which indicates from the indice number of the element to retrieve its
neighbourhood, how we have to store the given information into the final
result.
It is the main information when we are doing coarse-graining of information.

The main coding purpose of this module and this class is to standarize such
important information and process.

"""

import numpy as np
import warnings


class Map_Vals_i:
    """Class which maps the result to reference output value."""

    def _initialization(self):
        self.mapper = None
        self.n_in = None
        self.n_out = None
        self.collapse = False
        self.prefilter = lambda idx: idx

    def __init__(self, mapper, n_in=None, n_out=None, sptype='matrix'):
        """The map to the elements indices to the stored indices.

        Parameters
        ----------
        mapper
        n_in: int or None (default=None)
            the size of the input. If None, it is open.
        n_out: int or None (default=None)
            the size of the output. If None, it is open.
        sptype: str (default='matrix')
            the type of mapper.

        """
        self._initialization()
        self._format_mapper(mapper, n_in, n_out)
        self.sptype = sptype

    def __getitem__(self, key):
        """Apply mapper.

        Parameters
        ----------
        key: tuple
            it contains the feature retriever object, the indices of the
            elements and the perturbations indices.

        Returns
        -------
        map_key: int, list or np.ndarray
            the mapped indices.

        """
        featret_o, i, k = key
        if type(i) == int:
            i = self.prefilter(i)
            return [self.mapper(featret_o, i, k)]
        else:
            i = [self.prefilter(j) for j in i]
            return self.mapper(featret_o, i, k)

    def apply(self, o, i, k):
        """Apply the mapping.

        Parameters
        ----------
        key: tuple
        o: pst.FeatureManagement.FeaturesRetriever
            feature retriever object.
        i: int, list or np.ndarray
            the indices of the elements.
        k: int or list or np.ndarray
            the perturbations indices.

        Returns
        -------
        map_key: int, list or np.ndarray
            the mapped indices.

        """
        return self[o, i, k]

    def set_sptype(self, sptype):
        """Set type of mapping.

        Parameters
        ----------
        sptype: str
            type of mapping.

        """
        assert(type(sptype) == str)
        self.sptype = sptype

    def set_prefilter(self, filter_):
        """Set a function which filters the indice applied. It is useful
        for parallelization tasks.

        Parameters
        ----------
        filter_: int, slice, list or np.ndarray
            the pretransformation of the indice of the element.

        """
        if type(filter_) == int:
            self.prefilter = lambda i: i+filter_
        elif type(filter_) == slice:
            start, stop, step = filter_.start, filter_.stop, filter_.step
            start = 0 if start is None else start
            step = 1 if step is None else step
            self.prefilter = lambda i: range(start, stop, step)[i]
        elif type(filter_) in [np.ndarray, list]:
            self.prefilter = lambda i: int(filter_[i])

    def _format_mapper(self, mapper, n_in, n_out):
        """Format the mapper function.

        Parameters
        ----------
        mapper: int, float, list, tuple, np.ndarray, function or instance
            the main mapper information from elements indices to to-store
            indices.
        n_in: int
            the size of the input. If None, the inputs are open.
        n_out: int
            the size of the output. If None, the outputs are open.

        """
        if type(mapper) in [int, float, list, tuple]:
            if type(mapper) in [int, float]:
                mapper = int(mapper)
            else:
                mapper = [int(m) for m in mapper]
            self.mapper = lambda s, i, k: mapper
            self.n_out = 1
            self.n_in = n_in
        elif type(mapper) == np.ndarray:
            ## Transform indice
            mapper = mapper.astype(int)
            self.mapper = lambda s, idx, k: mapper[idx]
            self.n_in = len(mapper)
            self.n_out = len(np.unique(mapper))
        elif type(mapper).__name__ == 'function':
            self.mapper = mapper
            self.n_in = n_in
            self.n_out = n_out
        elif type(mapper).__name__ == 'instance':
            try:
                mapper[None, 0, 0]
            except:
                warnings.warn("Possibly incorrect function mapper input.")
            self.mapper = lambda s, i, k: mapper[s, i, k]
            self.n_in = n_in
            self.n_out = n_out


def create_mapper_vals_i(type_sp='correlation', features_out=None):
    """Create the values.

    Parameters
    ----------
    type_sp: str, tuple, np.ndarray, function, instance (default='correlation')
        the type of mapping.
    features_out: int, slice, np.ndarray (default=None)
        the features out information.

    Returns
    -------
    _map_vals_i: pst.Map_Vals_i
        the mapper instance from indices of elements to to-store indices.

    """
    mapper, n_in, n_out = None, None, None
    _map_vals_i = None

    ## 0. Preprocessing inputs
    try:
        if 'features' in dir(features_out):
            features_out = features_out.features[0]
    except:
        pass
    if type(type_sp) == tuple:
        if len(type_sp) == 2:
            n_in = len(features_out)
            n_out = type_sp[1]
        elif len(type_sp) == 3:
            n_in = type_sp[1]
            n_out = type_sp[2]
        type_sp = type_sp[0]
    if features_out is not None and n_in is None:
        if type(features_out) != slice:
            n_in = len(features_out)
    if features_out is not None:
        if type(features_out) == int:
            n_out = features_out
        elif type(features_out) == slice:
            n_vals_i = (features_out.stop+1-features_out.start)
            n_out = n_vals_i/features_out.step
        elif type(features_out) == np.ndarray:
            if len(features_out.shape) == 1:
                features_out = features_out.astype(int)
            else:
                features_out = features_out[:, 0].astype(int)
            ## Transform 2 indices
            mapper = -1*np.ones(len(features_out))
            for i in range(len(np.unique(features_out))):
                mapper[(features_out == np.unique(features_out)[i])] = i
            assert(np.sum(mapper == (-1)) == 0)
            n_in = len(features_out)
            n_out = len(np.unique(features_out).ravel())
#        elif type(features_out) not in [int, slice, np.ndarray]:
#            n_out = len(features_out)
    ## 1. Create mapper
    if type(type_sp) == str:
        if type_sp == 'correlation':
            if mapper is not None:
                _map_vals_i = Map_Vals_i(mapper, sptype="correlation")
            else:
                raise TypeError("Not enough information to build the mapper.")
        elif type_sp == 'matrix':
            funct = lambda self, idx, k: idx
            _map_vals_i = Map_Vals_i(funct, n_in, n_out)
    elif type(type_sp) == np.ndarray:
        ## Transform 2 indices
        corr_arr = -1*np.ones(len(type_sp))
        for i in range(len(np.unique(type_sp))):
            corr_arr[(type_sp == np.unique(type_sp)[i]).ravel()] = i
        assert(np.sum(corr_arr == (-1)) == 0)
        _map_vals_i = Map_Vals_i(corr_arr)
    elif type(type_sp).__name__ in ['function']:
        n_in = len(features_out) if features_out is not None else None
        _map_vals_i = Map_Vals_i(type_sp, n_in, n_out)
    elif type(type_sp).__name__ in ['instance']:
        _map_vals_i = type_sp
    if _map_vals_i is None:
        funct = lambda self, idx, k: idx
        _map_vals_i = Map_Vals_i(funct, n_in, n_out)
    return _map_vals_i
