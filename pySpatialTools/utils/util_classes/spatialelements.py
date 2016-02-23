
"""
Locations
---------


TODO
----
location parameter for each element

"""

import numpy as np
import warnings
from scipy.spatial.distance import cdist


class SpatialElementsCollection:
    """Class which stores spatial elements collections and provides a functions
    and utilites to perform main computations with them.
    """

    def __init__(self, sp_ele_collection, points_id=None):
        self.elements = self._format_elements(sp_ele_collection)
        self.points_id = self._format_tags(points_id)
        self.n_elements = len(self.elements)

    def __getitem__(self, i):
        if self.points_id is None:
            if self.n_points <= i or i < 0:
                raise IndexError("Index out of bonds")
            loc_i = self.locations[i, :]
        else:
            try:
                idx = np.where(self.points_id == i)[0][0]
                loc_i = self.locations[idx, :]
            except:
                raise IndexError("Index out of bonds")
        return loc_i

    def __iter__(self):
        for i in xrange(self.n_points):
            yield self[i]

    def __eq__(self, loc):
        "Retrieve which points have the same coordinates."
        logi = self.elements == loc
        return logi

    def relabel_points(self, relabel_map):
        "Relabel the elements."
        if self.points_id is None:
            if relabel_map == np.ndarray:
                self.points_id = relabel_map
            else:
                raise TypeError("Not correct input.")
        else:
            new_tags = -2 * np.ones(self.points_id.shape[0])
            for p in np.unique(self.points_id):
                if p in relabel_map.keys():
                    new_tags[self.points_id == p] = relabel_map[p]
            logi = new_tags == -2
            new_tags[logi] = self.points_id[logi]
            self.points_id = new_tags

    def _format_elements(self, sp_ele_collection):
        "Format properly the elements."
        if type(sp_ele_collection) == np.ndarray:
            return Locations(sp_ele_collection)
        else:
            try:
                len(sp_ele_collection)
            except:
                raise TypeError("Collection don't have __len__ function.")

    def _format_tags(self, tags):
        if tags is not None:
            if type(tags) in [np.ndarray, list]:
                tags = np.array(tags)
                if np.unique(tags).shape[0] != tags.shape[0]:
                    raise Exception("Non-unique labels for elements.")
                if tags.shape[0] != self.n_elements:
                    raise Exception("Not correct shape for tags.")
            else:
                raise TypeError("Not correct labels type for elements.")
        return tags


class Locations:
    """Class representing a locations object in which contains a locations
    matrix and utils for the package pySpatialTools.
    """

    def __init__(self, locations, points_id=None):
        locations = self._to_locations(locations)
        self.n_dim = locations.shape[1]
        self.n_points = locations.shape[0]
        self.locations = locations
        self.points_id = points_id

    def __eq__(self, loc):
        "Retrieve which points have the same coordinates."
        loc = self._to_loc(loc)
        logi = np.ones(self.n_points).astype(bool)
        for i_dim in range(self.n_dim):
            logi = np.logical_and(logi, self.locations[:, i_dim] == loc[i_dim])
        return logi

    def __iter__(self):
        for i in xrange(self.n_points):
            yield self[i]

    def __len__(self):
        return self.n_points

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise IndexError("Not possible to get collectively.")
        if self.points_id is None:
            if self.n_points <= i or i < 0:
                raise IndexError("Index out of bonds")
            loc_i = self.locations[i, :]
        else:
            try:
                idx = np.where(self.points_id == i)[0][0]
                loc_i = self.locations[idx, :]
            except:
                raise IndexError("Index out of bonds")
        return loc_i

    def _to_loc(self, loc):
        "Transform a location array in a locations"
        return loc.ravel()

    def _to_locations(self, locations):
        "Transform a locations array into the proper shape."
        if type(locations) == np.ndarray:
            if len(locations.shape) == 1:
                locations = locations.reshape((locations.shape[0], 1))
        return locations

    def compute_distance(self, loc, kwargs={}):
        """Compute distance between loc and all the other points in the
        collection.
        """
        ## Extend to other types of elements
        loc = self._to_loc(loc)
        loc = loc.reshape((1, self.n_dim))
        distances = cdist(loc, np.array(self.locations), **kwargs)
        return distances

    def _check_coord(self, i_locs):
        """Function to check if the input are coordinates or indices. The input
        is a coordinate when is an array with the same dimension that the pool
        of retrievable locations stored in retriever.data or in self.data.

        Parameters
        ----------
        i_locs: int, list of ints, numpy.ndarray or list of numpy.ndarray
            the locations information.

        Returns
        -------
        checker_coord: boolean
            if there are coordinates True, if there are indices False.

        """
        ## Get individuals
        if type(i_locs) == list:
            check_loc = i_locs[0]
        else:
            check_loc = i_locs
        ## Get checker
        if type(check_loc) in [int, np.int32, np.int64, np.ndarray]:
            if type(check_loc) != np.ndarray:
                checker_coord = False
            else:
                d_sh = self.locations.shape
                if len(check_loc.shape) == len(d_sh):
                    checker_coord = True
                else:
                    checker_coord = False
                    warnings.warn("Not correct shape for coordinates.")
        else:
            checker_coord = None
        return checker_coord

    @property
    def data(self):
        return self.locations

    def in_block_distance_d(self, loc, d):
        "If there is in less than d distance in all the dimensions."
        loc = self._to_loc(loc)
        loc = loc if len(loc.shape) == 2 else loc.reshape((1, len(loc)))
        logi = np.ones(len(self.locations))
        for i_dim in range(self.n_dim):
            dist_i = cdist(loc[:, [i_dim]], self.locations[:, [i_dim]])
            logi_i = dist_i < d
            logi = np.logical_and(logi, logi_i)
        return logi

    def in_manhattan_d(self, loc, d):
        """Compute if the location loc is in d manhattan distance of each of
        the points of this collection.
        """
        distances = self.compute_distance(loc, {'metric': 'cityblock'})
        logi = distances < d
        return logi

    def in_radio(self, loc, r, kwargs={}):
        "If there is in less than r distance in the metric selected."
        logi = self.compute_distance(loc, kwargs) < r
        return logi

    def space_transformation(self, method, params):
        """Space transformation. It calls to the transformation functions or
        classes of the pySpatialTools.
        """
        if type(method).__name__ == 'function':
            self.locations = method(self.locations, params)
        else:
            self.locations = method.apply_transformation(self.location,
                                                         **params)
