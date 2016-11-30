
"""
utils
-----
Utils for 2-dim spatial discretization.


TODO
----
Create voronoi diagram from line segments:
[1] http://gis.stackexchange.com/questions/104631/create-voronoi-diagram-from-
line-segments

"""

import numpy as np
import shapely
from shapely import ops
from scipy.spatial import Voronoi
from sklearn.neighbors import KDTree


def indices_assignation(indices, regions_id):
    """Function which acts to assign the indices obtained to the regions ids.

    Parameters
    ----------
    indices: array_like
        the indices of the renumbering.
    regions_id: array_like
        the regions id.

    Returns
    -------
    regions: array_like
        the regions assignation after renumbering.

    """
    n = len(indices)
    regions = -1*np.ones(n).astype(int)
    boolean = indices >= 0
    regions[boolean] = regions_id[indices[boolean]]
    return regions


def mask_application_grid(p, points):
    """Returns the index in which is placed the point.

    Parameters
    ----------
    p: float
        the point to assign grid number.
    points: array_like
        the points which define the limits of the intervals.

    Returns
    -------
    i: integer
        indice of which grid number is the assigned to a point `p`.

    """
    if p < points[0] or p > points[-1]:
        return -1
    for i in xrange(points.shape[0]-1):
        if p <= points[i+1]:
            return i


def compute_limital_polygon(limits):
    """Compute a polygon with the imformation given in limits.

    Parameters
    ----------
    limits: shapely.Polygon, tuple, list, np.ndarray
        the information to define the limits.

    Returns
    -------
    lims: shapely.Polygon
        the information to define the limits.

    """
    if type(limits) == shapely.geometry.polygon.Polygon:
        lims = limits
    elif type(limits) in [tuple, list, np.ndarray]:
        lims = shapely.geometry.Polygon(limits)
    return lims


def match_regions(polygons, regionlocs, n_dim=2):
    """

    Parameters
    ----------
    polygons: list or array_like
        the polygons information.
    regionlocs: array_like
        the location information of the regions.
    n_dim: integer
        the number of dimensions.

    Returns
    -------
    assign_r: array_like
        the assignated regions.
    """
    n = len(polygons)
    centroids = np.zeros((n, n_dim))
    for i in xrange(n):
        centroids[i, :] = np.array(polygons[i])
    ret = KDTree(regionlocs)
    assign_r = np.zeros(n).astype(int)
    for i in xrange(n):
        assign_r[i] = ret.query(centroids[[i]])[1][0]
    return assign_r


def tesselation(regionlocs):
    """Function to create a tesselation from the regionlocs.

    Parameters
    ----------
    regionlocs: array_like
        the spatial locations that define each region considered.

    Returns
    -------
    polygons: shapely.Polygon
        the polygon object which contains the information to define it as
        a polygon.

    """
    vor = Voronoi(regionlocs)
    lines = []
    for line in vor.ridge_vertices:
        if -1 not in line:
            lines.append(shapely.geometry.LineString(vor.vertices[line]))
    pols = ops.polygonize(lines)
    polygons = [poly for poly in pols]
    return polygons

#from pySpatialTools.Retrieve.Discretization.spatialdiscretizer import \
#    SpatialDiscretizor

#from polygondiscretization import fit_polygondiscretizer
#    def transform2polygondiscretizer(self):
#        discretizer = fit_polygondiscretizer(self.regionlocs, self.regions_id)
#        return discretizer

    ##################### Definition of particularities #######################
    ###########################################################################
#    def fit_spatialmodel(self, data):
#        """Fit regions from distribution of tagged points in space.
#        TODO: Tesselation.
#        """
#        self.regionlocs, self.borders = somefunction(data)


#from sklearn.neighbors import KDTree
#from scipy.spatial.distance import cdist
#from pythonUtils.parallel_tools import distribute_tasks
