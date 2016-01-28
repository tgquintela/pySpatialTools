
"""
Spatial Discretizor utilities
-----------------------------
Module which contains the classes for discretize space. The main function of
these clases are transform the location points to a region ids.
If the point do not belong to the region discretized, the function has to
return -1.
The clases also implement some util functions related.

Compulsary requisits
--------------------
For each instantiable son classes of spatial discretizers, are required the
next functions and parameters:
- map_loc2regionid (function)
- map_regionid2regionlocs (function)
- compute_limits (function)
- check_neighbours (function) [TOMOVE: regionmetrics?]
- compute_contiguity (function) [TOMOVE: regionmetrics?]
- get_nregs (function)
- get_regions_id (function)
- get_regionslocs (function)
- limits (parameter)
- borders (parameter)
- regionlocs (parameter)
- regions_id (parameter)

TODO
----
- Complete irregular discretizer.
- Retrieve only populated regions. (Renumerate populated regions)
- Assign regions to points.
- Multiple regions
- Multiple discretization types aggregated
- Compute contiguity using correlation measure
- 1-neigh discretizer for nd discretization
- nd-grid discretization

"""


import numpy as np
from pySpatialTools.Retrieve.Spatial_Relations import format_out_relations
#from scipy.spatial.distance import cdist
#from sklearn.neighbors import KDTree
#from pythonUtils.parallel_tools import distribute_tasks


class SpatialDiscretizor:
    """Spatial Discretizor object. This object performs a discretization of the
    spatial domain and it is able to do:
    - Assign a static predefined regions to each point.
    - Retrieve neighbourhood defined by static regions.

    """

    limits = None
    borders = None
    regionlocs = None
    regions_id = None
    regionretriever = None

    def retrieve_region(self, point_i, info_i, ifdistance=False):
        """Retrieve the region to which the points given belong to in this
        discretization.

        Parameters
        ----------
        point_i: numpy.ndarray, shape(n, m) or shape (n,)
            the point or points we want to retrieve their regions.
        info_i: numpy.ndarray, shape(n,)
            the special information in order to retrieve neighs and regions.
        ifdistance: bool
            True if we want the distance.

        Returns
        -------
        region: numpy.ndarray or int
            the region id of the given points.

        """
        if len(point_i.shape) == 1:
            point_i = point_i.reshape(1, point_i.shape[0])
        region = self.map_loc2regionid(point_i)
        return region

    def retrieve_neigh(self, point_i, locs):
        """Retrieve the neighs given a point using this discretization. Could
        be an internal retrieve if point_i is an index or an external retrieve
        if point_i it is not a point in locs (point_i is a coordinates).

        Parameters
        ----------
        point_i: numpy.ndarray
            the point location for which we want its neighbours using the given
            discretization.
        locs: numpy.ndarray, shape(n,)
            the location of the points we want to get the neighs of point_i.

        Returns
        -------
        logi: numpy.ndarray boolean
            the boolean array of which locs are neighs (are in the same region)
            of point_i.

        """
        regions = self.map_loc2regionid(locs)
        if type(point_i) == int:
            region = regions[point_i]
        else:
            region = self.map_loc2regionid(point_i)
        logi = self.check_neighbours(region, regions)
        return logi

    def discretize(self, locs):
        """Discretize locs given their region_id.

        Parameters
        ----------
        locs: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regions: numpy.ndarray
            the region_id of each locs for this discretization.

        """
        regions = self.map_loc2regionid(locs)
        return regions

    def belong_region(self, point, region_id=None):
        """Function to compute the belonging of some point to the region
        selected.

        Parameters
        ----------
        point: numpy.ndarray, shape(2,) or tuple or list
            the coordinates of the point we want to check its belonging to the
            selected region.
        region_id: int or None
            the region we want to check. If it is None we will check the whole
            region defined by the discretization.

        Returns
        -------
        boolean: bool
            the belonging to the selected region.

        """
        if region_id is None:
            boolean = self.map_loc2regionid(point) != -1
        else:
            boolean = self.map_loc2regionid(point) == region_id
        return boolean

    def get_contiguity(self, region_id=None, out_='sparse'):
        """Get the whole contiguity or the contiguos regions of a given region.

        Parameters
        ----------
        region_id: int or None
            the regions we want to get their contiguous regions. If it is None
            it is retrieved the whole map of contiguity.
        out_: optional ['sparse', 'list', 'network', 'sp_relations']
            how to present the results.

        Returns
        -------
        contiguity: list or list of lists
            the contiguous regions.

        """
        contiguity = self.compute_contiguity_geom(region_id)
        if region_id is None:
            contiguity = format_out_relations(contiguity, out_)
        return contiguity

    def get_limits(self, region_id=None):
        """Function to compute the limits of the region.

        Parameters
        ----------
        region_id: numpy.ndarray or int
            the regions id of the regions we want to get their limits. If it is
            None it is retrieved the limits of the whole discretized space.

        Returns
        -------
        limits: numpy.ndarray
            the limits with an specific ordering.

        """
        if region_id is None:
            limits = self.limits
        else:
            limits = self.compute_limits(region_id)
        return limits

    def map2agglocs(self, locs):
        ""
        pass

    def check_neighbours(self, region, regions):
        "Returns the ones with the same region which is required."
        logi = regions == region
        return logi

    def check_neighbours_multiple(self, region, regions):
        N_r = len(regions)
        logi = np.zeros(N_r).astype(bool)
        for i in xrange(N_r):
            logi[i] = region in regions[i]
        return logi

    def get_activated_regionlocs(self, locs, geom=True):
        regions = np.unique(self.map_loc2regionid(locs))
        if locs is True:
            regionlocs = self.map_regionid2regionlocs(regions)
        else:
            regionslocs = np.zeros((regions.shape[0], locs.shape[1]))
            for i in xrange(regions.shape[0]):
                regionslocs[i, :] = locs[regions[i] == regions, :].mean(0)
        return regionlocs, regions
