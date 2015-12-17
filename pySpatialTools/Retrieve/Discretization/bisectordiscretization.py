
"""
Bisector discretization
-----------------------
Module oriented to group all the classes and functions related to the
discretization of a space using bisector borders.

TODO
----
Compute contiguity_geom

"""

import numpy as np
from spatialdiscretizer import SpatialDiscretizor
from polygondiscretization import fit_polygondiscretizer
from sklearn.neighbors import KDTree

from spatial_utils import tesselation, match_regions
import shapely


class BisectorSpatialDisc(SpatialDiscretizor):
    """A method of defining regions by only giving the central points and
    using the bisector as a border.
    """

    def __init__(self, r_points, regions_id):
        """The bisector discretizor needs the regionlocs points and the
        region ids of these points.
        """
        self.regionlocs = r_points
        self.regionretriever = KDTree(r_points)
        self.regions_id = regions_id
        self.compute_limits()

    def map_loc2regionid(self, locs):
        """Discretize locs returning their region_id.

        Parameters
        ----------
        locs: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regions_id: numpy.ndarray
            the region_id of each locs for this discretization.

        """
        ## Use kd trees in order to retrieve nearest points.
        idxs = self.regionretriever.query(locs, 1, False)[:, 0]
        ## Replace the indices by the regions_id
        regions_id = self.region_id[idxs]
        return regions_id

    def map_regionid2regionlocs(self, regions_id):
        """Function which maps the regions ID to their most representative
        location.
        """
        ## 0. Variable needed
        if type(regions_id) == int:
            regions_id = np.array(regions_id)
        n = regions_id.shape[0]
        ## 1. Regionlocs computing
        regionlocs = np.zeros((n, self.regionlocs.shape[1]))
        for i in xrange(n):
            idx = np.where(self.regions_id == regions_id[i])[0][0]
            regionlocs[i, :] = self.regionlocs[idx, :]
        return regionlocs

    def map_locs2regionlocs(self, locs):
        "Map locations to regionlocs."
        ## Use kd trees in order to retrieve nearest points.
        idxs = self.regionretriever.query(locs, 1, False)[:, 0]
        ## Return regionlocs
        regionlocs = self.regionlocs[idxs, :]
        return regionlocs

    def compute_contiguity_geom(self, limits):
        "Compute which regions are contiguous and returns a graph."
        ## Obtain random points around all the r_points
        ## Compute the two nearest points with different region_id
        ## Remove repeated pairs
        return

    def compute_limits(self, region_id=None):
        if region_id is None:
            polygons = self.tesselation()
            whole = shapely.ops.unary_union(polygons)
            limits = np.array(whole.bounds).reshape((2, 2)).T
        else:
            polygons = self.tesselation()
            i_r = match_regions(polygons, self.regionlocs)
            regionsid = self.regions_id[i_r]
            p = polygons[np.where(regionsid == region_id)[0]]
            limits = np.array(p.bounds).reshape((2, 2)).T
        return limits

    def transform2polygondiscretizer(self):
        discretizer = fit_polygondiscretizer(self.regionlocs, self.regions_id)
        return discretizer

    def tesselation(self):
        polygons = tesselation(self.regionlocs)
        return polygons
