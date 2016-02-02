
"""
jitter
------
Jitter module to perturbe system in order of testing methods.

"""

import numpy as np


class Jitter:

    _stds = 0

    def __init(self, stds):
        stds = np.array(stds)

    def apply(self, coordinates):
        jitter_d = np.random.random(coordinates.shape)
        new_coordinates = np.multiply(self._stds, jitter_d)
        return new_coordinates
