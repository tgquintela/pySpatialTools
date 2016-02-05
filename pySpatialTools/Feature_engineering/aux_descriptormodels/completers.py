
"""
Completer functions
-------------------
This module contain possible functions to complete the final measure.

"""

import numpy as np


def null_completer(measure, global_info=None):
    "Do not change the measure."
    return measure


def weighted_completer(measure, global_info):
    """Weight the different results using the global info.
    It is REQUIRED that the global_info is an array of the same length as the
    measure.
    """
    global_info = global_info.ravel()
    assert len(measure) == len(global_info)
    if global_info is None:
        return measure
    global_info = global_info.reshape((len(global_info), 1, 1))
    measure = np.multply(measure, global_info)
    return measure
