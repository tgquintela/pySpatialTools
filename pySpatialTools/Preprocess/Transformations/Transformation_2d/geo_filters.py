
"""
Module to filter coordinates.

TODO
----

"""

import numpy as np


def check_in_square_area(coord, lim_points):
    """Check if the coordinates given are in the square region defined by the
    lim_points.

    Parameters
    ----------
    coord: array_like
        the coordinates of the points we want to check.
    lim_points: array_like shape(2, 2)
        the limit coordinates. The order is (x coordinate, y coordinate).

    Returns
    -------
    logi: boolean array_like
        the boolean result of the check.

    """

    logi = np.ones(coord.shape[0]).astype(bool)
    logi = np.logical_and(logi, coord[:, 0] >= lim_points[0, 0])
    logi = np.logical_and(logi, coord[:, 0] <= lim_points[0, 1])
    logi = np.logical_and(logi, coord[:, 1] >= lim_points[1, 0])
    logi = np.logical_and(logi, coord[:, 1] <= lim_points[1, 1])
    return logi
