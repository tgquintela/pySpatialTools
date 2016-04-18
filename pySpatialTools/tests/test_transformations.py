
"""
test transformation
-------------------

"""

import numpy as np
from itertools import product
from pySpatialTools.Transformations.Transformation_2d import\
    check_in_square_area, ellipsoidal_projection, radians2degrees,\
    degrees2radians, spheroidal_projection, general_projection


def test():
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
