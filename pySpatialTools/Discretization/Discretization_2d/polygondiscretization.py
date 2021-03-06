
"""
Polygon discretization
----------------------
Group which groups all the utilities of polygon based discretization.

TODO
----
Fit from distribution of points tagged with region types.

"""

import numpy as np
from shapely import ops
from shapely.geometry import Point
from utils import match_regions, tesselation

from ..metricdiscretizor import BaseMetricDiscretizor


############################### Polygon based #################################
###############################################################################
class IrregularSpatialDisc(BaseMetricDiscretizor):
    """Grid spatial discretization."""
    n_dim = 2
    multiple = None

    def __init__(self, polygons, regions_id=None):
        "Main information to built the regions."
        self.multiple = None
        self._initialization()
        self.borders = polygons
        if regions_id is None:
            self.regions_id = np.arange(len(polygons))
        else:
            self.regions_id = regions_id
        self.regionlocs = self.map_regionid2regionlocs()
        self.limits = self.compute_limits()

    ################################ Functions ################################
    ###########################################################################
    def map_loc2regionid(self, locs):
        """Discretize locs returning their region_id.

        Parameters
        ----------
        locs: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regionid: numpy.ndarray
            the region_id of each locs for this discretization.

        """
        regionid = -1*np.ones(locs.shape[0]).astype(int)
        for i in xrange(len(self.borders)):
            ## See what regions contain these points in their bounding boxes
            limits = np.array(self.borders[i].bounds).reshape((2, 2)).T
            logi_i = np.logical_and(locs[:, 0] >= limits[0, 0],
                                    locs[:, 0] < limits[0, 1])
            logi_i = np.logical_and(logi_i, locs[:, 1] >= limits[1, 0])
            logi_i = np.logical_and(logi_i, locs[:, 1] < limits[1, 1])
            ## Decide region
            for j in np.where(logi_i)[0]:
                if regionid[j] == -1:
                    if self.borders.intersects(Point(locs[j, :])):
                        regionid[j] = self.regions_id[i]
        return regionid

    def map_regionid2regionlocs(self, regions=None):
        """Function which maps the regions ID to their most representative
        location.

        Parameters
        ----------
        locs: np.ndarray, shape (n, 2)
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regionlocs: array_like, shape (n, 2)
            the region locations of the each assigned location region.

        """
        if regions is None:
            n_dim = np.array(self.borders[0].centroid).shape[0]
            n = len(self.borders)
            regionlocs = np.zeros((n, n_dim))
            for i in xrange(n):
                regionlocs[i, :] = np.array(self.borders[i].centroid)
        else:
            regionlocs = np.zeros((regions.shape[0],
                                   self.regionslocs.shape[1]))
            for i in xrange(regions.shape):
                idx = np.where(self.regions_id == regions[i])[0][0]
                regionlocs[i, :] = self.regionlocs[idx, :]
        return regionlocs

    def map_locs2regionlocs(self, locs):
        """Map locations to regionlocs.

        Parameters
        ----------
        locs: array_like, shape (n, 2)
            the locations.

        Returns
        -------
        regionlocs: array_like, shape (n, 2)
            the region locations of the each assigned location region.

        """
        regionid = self.map_loc2regionid(locs)
        regionlocs = self.map_regionid2regionlocs(regionid)
        return regionlocs

    def map2aggloc_spec(self, locs):
        agglocs = np.zeros(self.regionlocs.shape).astype(float)
        regions = self.map_loc2regionid(locs)
        # Average between all the locs circles
        u_regions = regions.unique()
        for i in xrange(u_regions.shape[0]):
            logi = (regions == u_regions[i])
            agglocs[i, :] = np.mean(locs[logi, :], axis=0)
        return agglocs

    def _compute_limits(self, region_id=None):
        """Compute the limits of the whole system or of a specific region_id.

        Parameters
        ----------
        region_id: integer or None (default)
            the region id information.

        Returns
        -------
        limits: array_like
            the limits information.

        """
        if region_id is None:
            whole = ops.cascaded_union(self.borders)
            limits = np.array(whole.bounds).reshape((2, 2)).T
        else:
            idxs = np.where(self.regions_id == region_id)[0]
            reg = ops.cascaded_union([self.borders[i] for i in idxs])
            limits = np.array(reg.bounds).reshape((2, 2)).T
        return limits

    ################################ TODO: ###############################
    ##########################################################################
    def compute_contiguity_geom(self, limits):
        "Compute which regions are contiguous and returns a graph."
        # TODO:
        raise Exception("Not implemented function yet.")
        ## Obtain random points around all the r_points
        ## Compute the two nearest points with different region_id
        ## Remove repeated pairs
        return


def fit_polygondiscretizer(regionlocs, regions_id):
    polygons = tesselation(regionlocs)
    i_r = match_regions(polygons, regionlocs)
    regionsid = regions_id[i_r]
    return IrregularSpatialDisc(polygons, regionsid)


"""
Properties to compute
---------------------
* Intelligent pairs
Ovelaps of the bounding boxes
* Disjoint discretization
Intelligent selection of pairs and polygon1.intersection(polygon2).area == 0
or polygon1.overlaps(polygon2)
* Contiguous_geom
Intelligent selection of pairs and polygon1.intersection(polygon2).area == 0

"""
