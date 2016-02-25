
"""
Testing preprocess module
-------------------------
functions to test preprocess module.
"""

import numpy as np
from pySpatialTools.Preprocess import remove_unknown_locations,\
    jitter_group_imputation


def test():
    logi = np.random.randint(0, 2, 100)
    locations = np.random.random((100, 2))
    groups = np.random.randint(0, 20, 100)
    remove_unknown_locations(locations, logi)

    jitter_group_imputation(locations, logi, groups)
