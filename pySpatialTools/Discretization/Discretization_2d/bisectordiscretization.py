
"""
Bisector discretization
-----------------------
Module oriented to group all the classes and functions related to the
discretization of a space using bisector borders.

"""

from shapely import ops
import numpy as np
from sklearn.neighbors import KDTree

from ..metricdiscretizor import MetricDiscretizor
from utils import tesselation, match_regions


class BisectorSpatialDisc(MetricDiscretizor):
    """A method of defining regions by only giving the central points and
    using the bisector as a border.
    """
    n_dim = 2
    multiple = False

    def __init__(self, r_points, regions_id):
        """The bisector discretizor needs the regionlocs points and the
        region ids of these points.
        """
        self._initialization()
        assert len(r_points) == len(regions_id)
        self.regionlocs = r_points
        self.regions_id = regions_id
        self._compute_limits()
        self.regionretriever = KDTree(r_points)

    def _map_loc2regionid(self, locations):
        """Discretize locations returning their region_id.

        Parameters
        ----------
        locations: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regions_id: numpy.ndarray
            the region_id of each location for this discretization.

        """
        ## 0. Use kd trees in order to retrieve nearest points.
        idxs = self.regionretriever.query(locations, 1, False)[:, 0]
        ## 1. Replace the indices by the regions_id
        regions_id = self.regions_id[idxs]
        return regions_id

    def _map_regionid2regionlocs(self, regions_id):
        """Function which maps the regions ID to their most representative
        location.
        """
        ## 0. Variable needed
        if type(regions_id) == int:
            regions_id = np.array([regions_id])
        n = regions_id.shape[0]
        ## 1. Regionlocs computing
        regionlocs = np.zeros((n, self.regionlocs.shape[1]))
        for i in xrange(n):
            idx = np.where(self.regions_id == regions_id[i])[0][0]
            regionlocs[i, :] = self.regionlocs[idx, :]
        return regionlocs

    def _map_locs2regionlocs(self, locs):
        "Map locations to regionlocs."
        ## 0. Use kd trees in order to retrieve nearest points.
        idxs = self.regionretriever.query(locs, 1, False)[:, 0]
        ## 1. Return regionlocs
        regionlocs = self.regionlocs[idxs, :]
        return regionlocs

    def _compute_contiguity_geom(self, limits):
        "Compute which regions are contiguous and returns a graph."
        ## TODO:
        raise Exception("Not implemented function yet.")
        ## Obtain random points around all the r_points
        ## Compute the two nearest points with different region_id
        ## Remove repeated pairs
        return

    def _compute_limits(self, region_id=None):
        """WARNING: probably not yet completely implemented."""
        if region_id is None:
            polygons = tesselation(self.regionlocs)
            whole = ops.unary_union(polygons)
            limits = np.array(whole.bounds).reshape((2, 2)).T
        else:
            polygons = tesselation(self.regionlocs)
            i_r = match_regions(polygons, self.regionlocs)
            regionsid = self.regions_id[i_r]
            p = polygons[np.where(regionsid == region_id)[0]]
            limits = np.array(p.bounds).reshape((2, 2)).T
        return limits
