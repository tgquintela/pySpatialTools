
"""
Testing preprocess module
-------------------------
functions to test preprocess module.
"""

import numpy as np
from pySpatialTools.Preprocess import remove_unknown_locations,\
    jitter_group_imputation, combinatorial_combination_features


def test():
    logi = np.random.randint(0, 2, 100)
    locations = np.random.random((100, 2))
    groups = np.random.randint(0, 20, 100)
    remove_unknown_locations(locations, logi)

    jitter_group_imputation(locations, logi, groups)

    sh = 10, 3
    cat_feats = np.random.randint(0, 4, np.prod(sh)).reshape(sh)
    combinatorial_combination_features(cat_feats)
