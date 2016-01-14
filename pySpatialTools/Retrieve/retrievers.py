
"""
Retrievers
----------
The objects to retrieve neighbours in flat (non-splitted) space.


Structure:
----------
- Point Retrievers
- Region Retrievers
| -- Retrieve regions, distances regions
| -- Retrieve points of regions, distances of points

TODO:
----
- Ifdistance better implementation
- Exclude better implementation
- Multiple regions
- Multiple points to get neighs
- Create a general elementRetriever (the desired Great Unification)

"""

#from scipy.spatial.distance import cdist
#from scipy.spatial import KDTree
#from sklearn.neighbors import KDTree
import numpy as np


class Retriever:
    """Class which contains the retriever of points.
    TODO
    ----
    - More information in the defintion of the retriever.
    """

    locs = None
    autolocs = True
    info_ret = None
    info_f = None
    relative_pos = None
    flag_auto = False
    retriever = None
    ifdistance = False

    ## TODO:
    __tags__ = None

    def set_locs(self, locs, info_ret):
        "Set locations for retrieving their neighs."
        self.locs = locs
        self.info_ret = info_ret

    def retrieve_neighs(self, i_loc, info_i={}, ifdistance=None):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        """
        ## 0. Prepare variables
        info_i = self.get_info_i(i_loc, info_i)
        ifdistance = self.ifdistance if ifdistance is None else ifdistance
        ## 1. Retrieve neighs
        neighs, dists = self.retrieve_neighs_spec(i_loc, info_i, ifdistance)
        ## 2. Exclude auto if it is needed
        neighs, dists = self.format_output(i_loc, neighs, dists)
        neighs, dists = np.array(neighs), np.array(dists)
        return neighs, dists

    ########################### Auxiliar functions ############################
    ###########################################################################
    def exclude_auto(self, i_loc, neighs, dists):
        "Exclude auto points if there exist in the neighs retrieved."
        ## 0. Detect input i_loc and retrieve to_exclude_points list
        if type(i_loc) in [int, np.int32, np.int64]:
            to_exclude_points = [i_loc]
        elif type(i_loc) == np.ndarray:
            ###########################################################
            to_exclude_points = self.build_excluded_points(i_loc)
            ###########################################################
        ## 1. Excluding task
        n_p = np.array(neighs).shape[0]
        idxs_exclude = [i for i in xrange(n_p) if neighs[i]
                        in to_exclude_points]
        neighs = [neighs[i] for i in xrange(n_p) if i not in idxs_exclude]
        if dists is not None:
            dists = [dists[i] for i in xrange(n_p) if i not in idxs_exclude]
        return neighs, dists

    def build_excluded_points(self, i_loc):
        "Build the excluded points from i_loc."
        sh = i_loc.shape
        i_loc = i_loc if len(sh) == 2 else i_loc.reshape(1, sh[0])
        logi = np.ones(self.retriever.data.shape[0]).astype(bool)
        for i in range(self.retriever.data.shape[1]):
            aux_logi = np.array(self.retriever.data)[:, i] == i_loc[:, i]
            logi = np.logical_and(logi, aux_logi)
        to_exclude_points = np.where(logi)[0]
        return to_exclude_points

    def check_coord(self, i_locs):
        """Function to check if the input are coordinates or indices. The input
        is a coordinate when is an array with the same dimension that the pool
        of retrievable locations stored in retriever.data.

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
                if len(check_loc.shape) == len(self.retriever.data.shape):
                    checker_coord = True
                else:
                    #raise Error
                    raise Exception()
                    pass
        else:
            checker_coord = None
        return checker_coord

    def get_info_i(self, i_loc, info_i):
        "Get information of retrieving point."
        if not info_i:
            if type(i_loc) in [int, np.int32, np.int64]:
                if type(self.info_ret) in [list, np.ndarray]:
                    info_i = self.info_ret[i_loc]
                else:
                    info_i = self.info_ret
            else:
                # raise Error if not set info_f
                if type(self.info_f).__name__ == 'name':
                    info_i = self.info_f(i_loc, info_i)
                else:
                    pass
        return info_i

    def get_loc_i(self, i_loc):
        "Get location."
        if type(i_loc) in [int, np.int32, np.int64]:
            if self.autolocs:
                if len(self.retriever.data.shape) == 1:
                    loc_i = np.array(self.retriever.data[i_loc])
                elif len(self.retriever.data.shape) == 2:
                    loc_i = np.array(self.retriever.data[i_loc, :])
                    loc_i = loc_i.reshape((1, loc_i.shape[0]))
            else:
                if len(self.locs.shape) == 1:
                    loc_i = self.locs[[i_loc]]
                else:
                    loc_i = self.locs[[i_loc], :]
        else:
            loc_i = np.array(i_loc)
        return loc_i
