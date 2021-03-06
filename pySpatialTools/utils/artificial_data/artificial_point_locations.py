
"""
artificial point locations
--------------------------
Module which groups all the functions related with artificial point locations
data.

"""

import numpy as np


def random_space_points(n_points, n_dim):
    """Uniformelly random sampled points from the space.

    Parameters
    ----------
    n_points: int
        the number of points.
    n_dim: int
        the number of the dimensions.

    Returns
    -------
    locs: np.ndarray
        the randomly generated points.

    """
    locs = np.random.random((n_points, n_dim))
    return locs


def random_transformed_space_points(n_points, n_dim, funct):
    """Random transformed space points. There are points sampled random from
    space uniformly and after that are transformed by the funct given.

    Parameters
    ----------
    n_points: int
        the number of points.
    n_dim: int
        the number of the dimensions.
    funct: function or list
        the functions we want to apply to the random generated points.

    Returns
    -------
    locs: np.ndarray
        the randomly generated points.

    """
    if funct is None:
        return random_space_points(n_points, n_dim)
    if type(funct) != list:
        funct = [funct]
    locs = np.random.random((n_points, n_dim))
    locs = np.vstack([f(locs) for f in funct]).T
    return locs
