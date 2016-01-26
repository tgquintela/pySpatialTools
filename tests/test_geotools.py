

## Imports
from pySpatialTools.Geo_tools import general_projection, check_in_square_area
import numpy as np

## Check spain
def test():
    coord = np.array([[-19., 26.], [-18., 26.], [-19., 28.], [-18., 28.]])

    lim_points = np.array([[-18.25, 4.5], [27.75, 44]])
    logi = check_in_square_area(coord, lim_points)

    coord2 = general_projection(coord, None, 'ellipsoidal', inverse=False, radians=False)
