
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


class Sp_DescriptorMapper:
    """Spatial descriptor mapper to indicate the path of possible options to
    compute spatial descriptors.
    """
    _mapper = lambda s, idx: (0, 0, 0, 0, 0, 0, 0, 0)
    __name__ = "pst.Sp_DescriptorMapper"

    def __init__(self, staticneighs=None, mapretinput=None, mapretout=None,
                 mapfeatinput=None, mapfeatoutput=None):

        dummymapper = lambda idx: 0

        if staticneighs is None:
            if type(staticneighs) == np.ndarray:
                staticneighs = lambda idx: staticneighs[idx]
            if type(staticneighs).__name__ == 'function':
                pass
            else:
                staticneighs = dummymapper

        if mapretinput is None:
            if type(mapretinput) == np.ndarray:
                mapretinput = lambda idx: mapretinput[idx]
            if type(mapretinput).__name__ == 'function':
                pass
            else:
                mapretinput = dummymapper

        if mapretout is None:
            if type(mapretout) == np.ndarray:
                mapretout = lambda idx: mapretout[idx]
            if type(mapretout).__name__ == 'function':
                pass
            else:
                mapretout = dummymapper

        if mapfeatinput is None:
            if type(mapfeatinput) == np.ndarray:
                mapfeatinput = lambda idx: mapfeatinput[idx]
            if type(mapfeatinput).__name__ == 'function':
                pass
            else:
                mapfeatinput = dummymapper

        if mapfeatoutput is None:
            if type(mapfeatoutput) == np.ndarray:
                mapfeatoutput = lambda idx: mapfeatoutput[idx]
            if type(mapfeatoutput).__name__ == 'function':
                pass
            else:
                mapfeatoutput = dummymapper

        self._mapper = lambda i: (staticneighs(i), mapretinput(i),
                                  mapretout(i), mapfeatinput(i),
                                  mapfeatoutput(i))

    def __getitem__(self, keys):
        if type(keys) == int:
            istatic, iret, irout, ifeat, ifout = self._mapper(keys)
        else:
            raise TypeError("Not correct input for spatial descriptor mapper.")
        return istatic, iret, irout, ifeat, ifout

    def set_from_array(self, array_mapper):
        "Set mapper from array."
        if array_mapper.shape[1] != 5:
            msg = "Not correct shape of array to be a spatial mapper."
            raise TypeError(msg)
        self._mapper = lambda idx: tuple(array_mapper[idx])

    def set_from_function(self, function_mapper):
        try:
            a, b, c, d, e = function_mapper(0)
            self._mapper = function_mapper
        except:
            raise TypeError("Incorrect function mapper.")

    def checker(self, constraints):
        "TODO: checker functions if this mapper selector fits the constraints."
        pass

    def set_default_with_constraints(self, constraints):
        "TODO: default builder of the but with constraints."
        pass
