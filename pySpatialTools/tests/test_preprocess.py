
"""
Testing preprocess module
-------------------------
functions to test preprocess module.
"""

import numpy as np
from itertools import product
from pySpatialTools.Preprocess import remove_unknown_locations,\
    jitter_group_imputation, combinatorial_combination_features
from pySpatialTools.Preprocess.Transformations.Transformation_2d import\
    check_in_square_area, ellipsoidal_projection, radians2degrees,\
    degrees2radians, spheroidal_projection, general_projection


def test():
    logi = np.random.randint(0, 2, 100)
    locations = np.random.random((100, 2))
    groups = np.random.randint(0, 20, 100)
    remove_unknown_locations(locations, logi)

    jitter_group_imputation(locations, logi, groups)

    sh = 10, 3
    cat_feats = np.random.randint(0, 4, np.prod(sh)).reshape(sh)
    combinatorial_combination_features(cat_feats)

    ###########################################################################
    ############################# TRANSFORMATION ##############################
    ###########################################################################
    coord = np.random.random((100, 2))*2
    lim_points = np.array([[0., 1.], [0., 1.]])
    check_in_square_area(coord, lim_points)
    radians2degrees(coord)
    degrees2radians(coord)
    ## Assert inverse proper definition
    coord_i = spheroidal_projection(spheroidal_projection(coord), True)
    np.testing.assert_array_almost_equal(coord, coord_i)
    ## TODO: revise that
#    coord_i = ellipsoidal_projection(ellipsoidal_projection(coord), True)
#    np.testing.assert_array_almost_equal(coord, coord_i)

    pos = [['spheroidal', 'ellipsoidal'], [True, False], [True, False]]
    for p in product(*pos):
        general_projection(coord, method=p[0], inverse=p[1], radians=p[2])
