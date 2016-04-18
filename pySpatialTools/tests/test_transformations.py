
"""
test transformation
-------------------

"""

import numpy as np
from pySpatialTools.Transformations.Transformation_2d import\
    check_in_square_area, ellipsoidal_projection, radians2degrees,\
    degrees2radians, spheroidal_projection


def test():
    coord = np.random.random((100, 2))*2
    lim_points = np.array([[0., 1.], [0., 1.]])
    check_in_square_area(coord, lim_points)
    radians2degrees(coord)
    degrees2radians(coord)
    spheroidal_projection(coord)
    ellipsoidal_projection(coord)
