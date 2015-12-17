
"""
Spatial utils
-------------
Module which groups the functions related with spatial utils which can be
useful.


TODO
----
Create voronoi diagram from line segments:
[1] http://gis.stackexchange.com/questions/104631/create-voronoi-diagram-from-
line-segments


"""

import numpy as np
import shapely
from scipy.spatial import Voronoi
from sklearn.neighbors import KDTree


def indices_assignation(indices, regions_id):
    "Function which acts to assign the indices obtained to the regions ids."
    n = indices.shape[0]
    regions = -1*np.ones(n).astype(int)
    boolean = indices >= 0
    regions[boolean] = regions_id[indices[boolean]]
    return regions


def mask_application(p, points):
    "Returns the index in which is placed the point."
    if p < points[0] or p > points[-1]:
        return -1
    for i in xrange(points.shape[0]-1):
        if p <= points[i+1]:
            return i


def compute_limital_polygon(limits):
    "Compute a poligon with the imformation given in limits."
    if type(limits) == shapely.geometry.polygon.Polygon:
        pass
    elif type(limits) in [tuple, list, np.ndarray]:
        lims = shapely.geometry.Polygon(limits)

    return lims


def match_regions(polygons, regionlocs, n_dim=2):
    n = len(polygons)
    centroids = np.zeros((n, n_dim))
    for i in xrange(n):
        centroids[i, :] = np.array(polygons[i])
    ret = KDTree(regionlocs)
    assign_r = np.zeros(n).astype(int)
    for i in xrange(n):
        assign_r[i] = ret.query(centroids[i, :])[1][0]
    return assign_r


def tesselation(regionlocs):
    vor = Voronoi(regionlocs)
    lines = []
    for line in vor.ridge_vertices:
        if -1 not in line:
            lines.append(shapely.geometry.LineString(vor.vertices[line]))
    pols = shapely.ops.polygonize(lines)
    polygons = [poly for poly in pols]
    return polygons
