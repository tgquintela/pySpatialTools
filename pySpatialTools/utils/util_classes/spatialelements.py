
"""
Spatial Elements
----------------
It is the module which groups all the classes that could store spatial
elements collections and procure functions to manage them properly.

"""

import numpy as np
import warnings
from scipy.spatial.distance import cdist


class SpatialElementsCollection:
    """Class which stores spatial elements collections and provides a functions
    and utilites to perform main computations with them.
    """

    def __init__(self, sp_ele_collection, elements_id=None):
        """The spatial elements collection instanciation.

        Parameters
        ----------
        sp_ele_collection: list or np.ndarray
            the collection of elements.
        elements_id: np.ndarray, list
            the tags of each element.

        """
        self.elements = self._format_elements(sp_ele_collection)
        self.n_elements = len(self.elements)
        self.elements_id = self._format_tags(elements_id)

    def __getitem__(self, i):
        """Get items of the elements `i`.

        Parameters
        ----------
        i: int
            the element `i` we want to get their locations.

        Returns
        -------
        ele_i: np.ndarray
            the spatial infomraion associated with the elements `i`.

        """
        if self.elements_id is None:
            if len(self) <= i or i < 0:
                raise IndexError("Index out of bonds")
            ele_i = self.elements[i]
        else:
            try:
                idx = np.where(self.elements_id == i)[0][0]
                ele_i = self.elements[idx]
            except:
                raise IndexError("Index out of bonds")
        return ele_i

    def __len__(self):
        """Number of total elements stored."""
        return len(self.elements)

    def __iter__(self):
        """Sequencial retrieving the locations stored in data.

        Returns
        -------
        loc_i: optional
            location of element `i`.

        """
        if self.elements_id is None:
            for i in xrange(len(self)):
                yield self[i]
        else:
            for e in self.elements_id:
                yield self[e]

    def __eq__(self, loc):
        """Retrieve which points have the same coordinates. It returns a
        1d-boolean array.

        Parameters
        ----------
        loc: optional
            the spatial information of some element.

        Returns
        -------
        logi: boolean np.ndarray
            the elements that have the same spatial information.

        """
        if type(self.elements) == list:
            logi = np.array([np.all(e == loc) for e in self.elements])
        else:
            logi = (self.elements == loc).ravel()
        return logi

    def relabel_elements(self, relabel_map):
        """Relabel the elements.

        Parameters
        ----------
        relabel_map: dict or np.ndarray
            the information to change the labels.

        """
        if self.elements_id is None:
            if type(relabel_map) == np.ndarray:
                self.elements_id = relabel_map
            else:
                raise TypeError("Not correct input.")
        else:
            new_tags = -2 * np.ones(len(self.elements_id))
            for p in np.unique(self.elements_id):
                if p in relabel_map.keys():
                    new_tags[self.elements_id == p] = relabel_map[p]
            logi = new_tags == -2
            new_tags[logi] = self.elements_id[logi]
            self.elements_id = new_tags

    def _format_elements(self, sp_ele_collection):
        """Format properly the elements.

        Parameters
        ----------
        sp_ele_collection: list or np.ndarray
            the collection of elements.

        """
        if type(sp_ele_collection) == np.ndarray:
            return Locations(sp_ele_collection)
        else:
            try:
                len(sp_ele_collection)
                return sp_ele_collection
            except:
                raise TypeError("Collection don't have __len__ function.")

    def _format_tags(self, tags):
        """Format tag.

        Parameters
        ----------
        tags: np.ndarray, list
            the tags we want to format.

        Returns
        -------
        tags: np.ndarray, list
            the tags formatted.

        """
        if tags is not None:
            if type(tags) in [np.ndarray, list]:
                tags = np.array(tags)
                if len(np.unique(tags)) != len(tags):
                    raise Exception("Non-unique labels for elements.")
                if len(tags) != self.n_elements:
                    raise Exception("Not correct length for tags.")
            else:
                raise TypeError("Not correct labels type for elements.")
        return tags


class Locations:
    """Class representing a locations object in which contains a locations
    matrix and utils for the package pySpatialTools.
    """

    def __init__(self, locations, points_id=None):
        """The locations object which stores the whole data.

        Parameters
        ----------
        locations: np.ndarray
            the locations to be stored.
        points_id: list or np.ndarray
            the points code id.

        """
        locations = self._to_locations(locations)
        self.n_dim = locations.shape[1]
        self.n_points = len(locations)
        self.locations = locations
        self.points_id = self._format_tags(points_id)

    def __eq__(self, loc):
        """Retrieve which points have the same coordinates.

        Parameters
        ----------
        loc: np.ndarray
            the location selected we want to get from data.

        Returns
        -------
        logi: boolean np.ndarray
            if the elements are equal to the input location.

        """
        loc = self._to_loc(loc)
        logi = np.ones(self.n_points).astype(bool)
        for i_dim in range(self.n_dim):
            logi = np.logical_and(logi, self.locations[:, i_dim] == loc[i_dim])
        return logi

    def __iter__(self):
        """Sequencial retrieving the locations stored in data.

        Returns
        -------
        loc_i: np.ndarray
            location of element `i`.

        """
        for i in xrange(self.n_points):
            yield self[i]

    def __len__(self):
        """Number of total points stored."""
        return self.n_points

    def __getitem__(self, i):
        """Get the item of element `i`.

        Parameters
        ----------
        i: int
            the element `i` we want to get their locations.

        Returns
        -------
        loc_i: np.ndarray
            location of element `i`.

        """
        if isinstance(i, slice):
            raise IndexError("Not possible to get collectively.")
        if self.points_id is None:
            if self.n_points <= i or i < 0:
                raise IndexError("Index out of bonds")
            loc_i = self.locations[i, :]
        else:
            idx = np.where(self.points_id == i)[0][0]
            loc_i = self.locations[idx, :]
        return loc_i

    def _format_tags(self, tags):
        """Format tag.

        Parameters
        ----------
        tags: np.ndarray, list
            the tags we want to format.

        Returns
        -------
        tags: np.ndarray, list
            the tags formatted.

        """
        if tags is not None:
            if type(tags) in [np.ndarray, list]:
                tags = np.array(tags)
                if len(np.unique(tags)) != len(tags):
                    raise Exception("Non-unique labels for elements.")
                if len(tags) != self.n_points:
                    raise Exception("Not correct length for tags.")
            else:
                raise TypeError("Not correct labels type for elements.")
        return tags

    def _to_loc(self, loc):
        """Transform a location array in a locations

        Parameters
        ----------
        loc: np.ndarray
            the location to be formatted properly as location.

        Returns
        -------
        loc: np.ndarray
            the location to be formatted properly as location.

        """
        return loc.ravel()

    def _to_locations(self, locations):
        """Transform a locations array into the proper shape.

        Parameters
        ----------
        locations: np.ndarray
            the locations to be formatted properly as locations.

        Returns
        -------
        locations: np.ndarray
            the locations to be formatted properly as locations.

        """
        if type(locations) == np.ndarray:
            if len(locations.shape) == 1:
                locations = locations.reshape((len(locations), 1))
        return locations

    def compute_distance(self, loc, kwargs={}):
        """Compute distance between loc and all the other points in the
        collection.

        Parameters
        ----------
        loc: np.ndarray
            the location to compute the distance to all the locations stored.

        Returns
        -------
        distances: np.ndarray
            the distances to all the locations stored from the location input.

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
                n_ch = len(check_loc.shape) == 1
                sh = (1, len(check_loc))
                check_loc = check_loc.reshape(sh) if n_ch == 1 else check_loc
                if check_loc.shape[1] == self.locations.shape[1]:
                    checker_coord = True
                else:
                    checker_coord = False
                    warnings.warn("Not correct shape for coordinates.")
        else:
            checker_coord = None
        return checker_coord

    @property
    def data(self):
        """The locations data stored."""
        return self.locations

    def in_block_distance_d(self, loc, d):
        """If there is in less than d distance in all the dimensions.

        Parameters
        ----------
        loc: np.ndarray
            the location selected as centroid.
        d: float
            the distance magnitude.

        Returns
        -------
        logi: boolean np.ndarray
            if the elements are in the distance of the selected location.

        """
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

        Parameters
        ----------
        loc: np.ndarray
            the location selected as centroid.
        d: float
            the distance magnitude.

        Returns
        -------
        logi: boolean np.ndarray
            if the elements are in the distance of the selected location.

        """
        distances = self.compute_distance(loc, {'metric': 'cityblock'})
        logi = distances < d
        return logi

    def in_radio(self, loc, r, kwargs={}):
        """If there is in less than r distance in the metric selected.

        Parameters
        ----------
        loc: np.ndarray
            the location selected as centroid.
        r: float
            the radius we want to check.
        kwargs: dict (defeault={})
            the parameters required by the compute_distance function.

        Returns
        -------
        logi: boolean np.ndarray
            if the elements are in the radius of the selected location.

        """
        logi = self.compute_distance(loc, kwargs) < r
        return logi

    def space_transformation(self, method, params):
        """Space transformation. It calls to the transformation functions or
        classes of the pySpatialTools.

        Parameters
        ----------
        method: function or instance
            the method of transformation we want to apply.
        params: dictionary
            the parameters required for the selected transformation.

        """
        if type(method).__name__ == 'function':
            self.locations = method(self.locations, params)
        else:
            self.locations = method.apply_transformation(self.locations,
                                                         **params)
