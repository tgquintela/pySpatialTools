
"""
Module to filter coordinates.
"""

import numpy as np


def check_correct_spain_coord(coord, radians=False):
    "Check if the coordinates given are in Spain or not."
    coord = np.array(coord)
    lim_points = np.array([[-18.25, 4.5], [27.75, 44]])
    if radians:
        lim_points = np.pi/180*lim_points

    logi = np.ones(coord.shape[0]).astype(bool)
    logi = np.logical_and(logi, coord[:, 0] >= lim_points[0, 0])
    logi = np.logical_and(logi, coord[:, 0] <= lim_points[0, 1])
    logi = np.logical_and(logi, coord[:, 1] >= lim_points[1, 0])
    logi = np.logical_and(logi, coord[:, 1] <= lim_points[1, 1])

    return logi


def filter_uncorrect_coord_spain(data, coord_vars, radians=False):
    "Filter not corrrect spain coordinates."
    coord = data[coord_vars].as_matrix()
    logi = check_correct_spain_coord(coord, radians)
    return data[logi]


def filter_bool_uncorrect_coord_spain(data, coord_vars, radians=False):
    "Filter data from pandas dataframe structure."
    coord = data[coord_vars].as_matrix()
    logi = check_correct_spain_coord(coord, radians)
    return logi
