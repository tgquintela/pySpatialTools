
"""
"""

import numpy as np


class General1_1Mapper:

    mapper = None
    n_in = None
    n_out = None

    def __init__(self, mapper, n_in=None, n_out=None):
        self._format_mapper(mapper, n_in, n_out)

    def __getitem__(self, key):
        return self.mapper(key)

    def _format_mapper(self, mapper, n_in, n_out):
        if type(mapper) in [int, float, list, tuple]:
            self.mapper = lambda x: mapper
            self.n_out = 1
            self.n_in = n_in
        elif type(mapper) == np.ndarray:
            self.mapper = lambda idx: mapper[idx]
            self.n_in = len(mapper)
            self.n_out = len(np.unique(mapper))
        elif type(mapper).__name__ == 'function':
            try:
                mapper(0)
            except:
                raise TypeError("Not correct function mapper input.")
            self.mapper = mapper
            self.n_in = n_in
            self.n_out = n_out
        elif type(mapper).__name__ == 'instance':
            try:
                mapper[0]
            except:
                raise TypeError("Not correct function mapper input.")
            self.mapper = lambda x: mapper[x]
            self.n_in = n_in
            self.n_out = n_out
