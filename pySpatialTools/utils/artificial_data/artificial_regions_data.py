
"""
Artificial regions data
-----------------------
Regions randomly defined.

"""

import numpy as np
from shapely.geometry import Polygon

numbertypes = [int, float, np.float, np.int32, np.int64]


def random_shapely_polygons(n_poly, bounding=(None, None), n_edges=0):
    """Generate a list of shapely polygons.

    Parameters
    ----------
    n_poly: int
        the number of polygons we want to generate.
    bounding: tuple (default=(None, None))
        the boundary limits.
    n_edges: int (default=0)
        the number of edges.

    Returns
    -------
    polygons: list of shapely.geometry.Polygon
        the polytions

    """
    polygons = []
    n_edges = [n_edges]*n_poly if type(n_edges) != list else n_edges
    for i in range(n_poly):
        polygons.append(random_shapely_polygon(bounding, n_edges[i]))
    return polygons


def random_shapely_polygon(bounding=(None, None), n_edges=0):
    """Generate a random polygon with different edges into a bounding box
    defined by the user.

    Parameters
    ----------
    bounding: tuple (default=(None, None))
        the boundary limits.
    n_edges: int (default=0)
        the number of edges.

    Returns
    -------
    polygon: shapely.geometry.Polygon
        the polytion we want to randomly generate.

    """
    ## Ensure correct dimensions
    assert(len(bounding) in [2, 3])
    n_dim = len(bounding)
    bounding = list(bounding)
    ## Define the bounding box properly
    for dim in range(len(bounding)):
        if bounding[dim] is not None:
            assert(len(bounding[dim]) == 2)
            assert(type(bounding[dim][0]) in numbertypes)
            assert(type(bounding[dim][1]) in numbertypes)
        else:
            bounding[dim] = [0, 1]

    n_edges = np.random.randint(3, 100) if n_edges < 3 else n_edges
    edges = []
    for i in range(n_edges):
        edge = []
        for i in range(n_dim):
            r = np.random.random()
            r = r*(bounding[dim][1]-bounding[dim][0]) + bounding[dim][0]
            edge.append(r)
        edges.append(edge)
    polygon = Polygon(edges)
    return polygon
