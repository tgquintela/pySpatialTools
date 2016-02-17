
"""
Mapper vals i
-------------
Module which contains mapper for vals i

"""


import numpy as np
import warnings


class Map_Vals_i:
    "Class which maps the result to reference output value."

    def _initialization(self):
        self.mapper = None
        self.n_in = None
        self.n_out = None
        self.collapse = False

    def __init__(self, mapper, n_in=None, n_out=None):
        self._initialization()
        self._format_mapper(mapper, n_in, n_out)

    def __getitem__(self, key):
        featret_o, i, k = key
        if type(i) == int:
            return [self.mapper(featret_o, i, k)]
        else:
            return self.mapper(featret_o, i, k)

    def apply(self, o, i, k):
        return self[o, i, k]

    def reduce_parallel(self, elements):
        listtype = [type(e) == list for e in elements]
        if self.collapse:
            if listtype:
                pass
            else:
                pass
        else:
            if listtype:
                pass
            else:
                pass

    def _format_mapper(self, mapper, n_in, n_out):
        if type(mapper) in [int, float, list, tuple]:
            self.mapper = lambda s, i, k: mapper
            self.n_out = 1
            self.n_in = n_in
        elif type(mapper) == np.ndarray:
            ## Transform indice
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
    """Create the values."""
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
        n_in = len(features_out)
    if features_out is not None:
        if type(features_out) == int:
            n_out = features_out
        elif type(features_out) == slice:
            n_vals_i = (features_out.stop+1-features_out.start)
            n_out = n_vals_i/features_out.step
        elif type(features_out) == np.ndarray:
            mapper = features_out[:, 0].astype(int)
            n_in = len(features_out)
            n_out = len(np.unique(features_out).ravel())
        elif type(features_out) not in [int, slice, np.ndarray]:
            n_out = len(features_out)

    ## 1. Create mapper
    if type(type_sp) == str:
        if type_sp == 'correlation':
            if mapper is not None:
                _map_vals_i = Map_Vals_i(mapper)
            else:
                raise TypeError("Not enough information to build the mapper.")
        elif type_sp == 'matrix':
            funct = lambda self, idx, k: idx
            _map_vals_i = Map_Vals_i(funct, n_in, n_out)
    elif type(type_sp) == np.ndarray:
            _map_vals_i = Map_Vals_i(type_sp)
    elif type(type_sp).__name__ in ['function']:
        n_in = len(features_out) if features_out is not None else None
        _map_vals_i = Map_Vals_i(type_sp, n_in, n_out)
    elif type(type_sp).__name__ in ['instance']:
        _map_vals_i = type_sp
    if _map_vals_i is None:
        funct = lambda self, idx, k: idx
        _map_vals_i = Map_Vals_i(funct, n_in, n_out)
    return _map_vals_i

#        if maps_vals_i is None:
#            self._maps_vals_i = Map_Vals_i(lambda self, i, k=0: i)
#        else:
#            if type(maps_vals_i).__name__ == 'function':
#                self._maps_vals_i = Map_Vals_i(lambda self, i, k=0: i)
#            else:
#                self._maps_vals_i = Map_Vals_i(maps_vals_i)
