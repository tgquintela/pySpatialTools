
"""
Spatial Discretizor
-------------------
Module which contains the classes to 'discretize' a topological space.
When we talk about discretize we talk about creating a non-bijective
mathematical application between two topological spaces.

The main function of an spatial discretization class is the transformation of
the spatial information of one element in some topological space to the spatial
information in another topological space in which each tological elements
contains a group of topological elements of the previous topological space.

If the point do not belong to the space discretized, the function has to
return -1.
The clases also implement some useful functions related with this task.


Conventions
-----------
- Region_id as int, regionslocs as numpy.ndarray (even if it is an int)

TODO
----
- Complete irregular discretizer.
- Assign regions to points.
- Multiple regions
- nd-grid discretization



- Retrieve only populated regions. (Renumerate populated regions)
- Multiple discretization types aggregated
- Compute contiguity using correlation measure

"""


import numpy as np
import warnings
from utils import check_discretizors, check_flag_multi


class SpatialDiscretizor:
    """
    Spatial Discretizor object. This object performs a discretization of the
    spatial domain and it is able to do:
    - Assign a static predefined regions to each point.
    - Retrieve neighbourhood defined by static regions.
    This class acts as a base of all possible discretizers.

    """

    def __len__(self):
        """Returns the number of regions or discretization units."""
        return np.unique(self.regions_id)

    def __getitem__(self, key):
        """Get the regions_id which match with the input."""
        if type(key) == int:
            return self.regions_id[key]
        else:
            return self.discretize(key)

    def _initialization(self):
        """Function to initialize useful class parameters for discretizers."""
        self.limits = None
        self.borders = None
        self.regionlocs = None
        self.regions_id = None
        check_discretizors(self)

    def retrieve_region(self, element_i, info_i, ifdistance=False):
        """Retrieve the region to which the points given belong to in this
        discretization.
        **warning** it is in format retriever to be used in that way if the
        user consider in that way.

        Parameters
        ----------
        element_i: numpy.ndarray, shape(n, m) or shape (n,)
            the point or points we want to retrieve their regions.
        info_i: optional [ignored]
            the special information in order to retrieve neighs and regions.
        ifdistance: bool
            True if we want the distance.

        Returns
        -------
        region: numpy.ndarray or int
            the region id of the given points.

        See also
        --------
        pySpatialTools.Retrieve

        """
        region = self.discretize(element_i)
        return region

    def retrieve_neigh(self, element_i, elements):
        """Retrieve the neighs given a point using this discretization. Could
        be an internal retrieve if element_i is an index or an external
        retrieve if element_i it is not a point in elements (element_i is a
        coordinates).

        Parameters
        ----------
        element_i: numpy.ndarray
            the point location for which we want its neighbours using the given
            discretization.
        elements: optional
            the spatial information of the elements from which we want to get
            the neighs of element_i.

        Returns
        -------
        logi: numpy.ndarray boolean
            the boolean array of which elements are neighs (are in the same
            region) of element_i.

        """
        region = self.discretize(element_i)
        regions = self.discretize(elements)
        logi = self.check_neighbors(region, regions)
        return logi

    def discretize(self, elements):
        """Discretize elements given their region_id.

        Parameters
        ----------
        elements: optional
            the spatial information of the elements for which we want to obtain
            their region given that discretization.

        Returns
        -------
        regions: numpy.ndarray of int
            the region_id of each elements for this discretization.

        """
        regions = self._map_loc2regionid(elements)
        return regions

    def belong_region(self, elements, region_id=None):
        """Function to compute the belonging of some elements to the regions
        selected.

        Parameters
        ----------
        elements: optional
            the coordinates of the elements we want to check its belonging to
            the selected region.
        region_id: int or None
            the region we want to check. If it is None we will check the whole
            region defined by the discretization.

        Returns
        -------
        boolean: bool
            the belonging to the selected region.

        """
        if region_id is None:
            regions = self.discretize(elements)
            boolean = not self.check_neighbors(regions, -1)
        else:
            regions = self.discretize(elements)
            boolean = self.check_neighbors(regions, region_id)
        return boolean

    def get_contiguity(self, region_id=None, *params):
        """Get the whole contiguity or the contiguos regions of a given region.

        Parameters
        ----------
        region_id: int or None
            the regions we want to get their contiguous regions. If it is None
            it is retrieved the whole map of contiguity.
        params: list or tuple
            the instructions of which considerations we need to compute the
            contiguity we want.

        Returns
        -------
        contiguity: list or list of lists
            the contiguous regions.

        """
        contiguity = self._compute_contiguity_geom(region_id, *params)
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
        ## Check limits function
        def check_limits(limits):
            try:
                assert len(limits.shape) == 1
                assert len(limits) == self.n_dim * 2
                return True
            except:
                try:
                    assert len(limits.shape) == self.n_dim
                    assert limits.shape == tuple([2]*self.n_dim)
                    return True
                except:
                    return False
        ## Compute limits
        if region_id is None:
            limits = self.limits
        else:
            limits = self._compute_limits(region_id)
        ## Check output
        if not check_limits(limits):
            raise TypeError("Incorrect computation of limits.")
        return limits

    def get_activated_regions(self, elements, geom=True):
        """Get the regions that have at least one of the elements input in
        them.

        Parameters
        ----------
        elements: optional
            the spatial information of the elements for which we want to obtain
            their region given that discretization.

        Returns
        -------
        regions: numpy.ndarray
            the regions which have some elements in them.

        """
        if self.multiple:
            discretized = self.discretize(elements)
            ## for weighted
            if type(discretized) == tuple:
                discretized = discretized[0]
            regions = []
            for e in discretized:
                regions += list(e)
            regions = np.unique(regions)
        else:
            regions = np.unique(self.discretize(elements))
        return regions

    def check_neighbors(self, regions, region):
        """Check if the region is in each of the regions of pre-discretized
        list of elements.

        Parameters
        ----------
        regions: int or numpy.ndarray
            the regions id we want to check if there are similar to region.
        region: list or numpy.array
            the assignation of regions.

        Returns
        -------
        logi: numpy.ndarray boolean
            the boolean array of which have region coincidence (are in the same
            region).

        """
        ## Check if there is flag multi
        flag_multi = check_flag_multi(regions)
        if self.multiple:
            logi = self._check_neighbors_multiple(regions, region)
        else:
            msg = "Regions multi-assigned in a not multi-assign discretizor."
            if flag_multi:
                warnings.warn(msg)
                logi = self._check_neighbors_multiple(regions, region)
            else:
                logi = self._check_neighbors_individual(regions, region)
        return logi

    ###########################################################################
    ########################### Auxiliar functions ############################
    ###########################################################################
    def _check_neighbors_individual(self, region, regions):
        """Check if there is equal regions in a uni-assigned regions. Returns
        a boolean array."""
        logi = regions == region
        return logi

    def _check_neighbors_multiple(self, region, regions):
        """Check if there is equal regions in a multi-assigned regions. Returns
        a boolean array."""
        N_r = len(regions)
        logi = np.zeros(N_r).astype(bool)
        for i in xrange(N_r):
            logi[i] = region in regions[i]
        return logi
