
"""
General interpolation
---------------------
Module which contains the general functions and classes to interpolate features
distributed in a space.

"""

import numpy as np


def general_interpolate(features, points, method, kwargs):
    """General function which acts as a switcher to interpolate features
    distributed in the space.

    """

    if method == "":
        interpolation = np.zeros((points.shape[0], features.shape[1]))

    return interpolation
