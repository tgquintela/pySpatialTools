
"""
Point retriever
---------------
Point retriever module in which is stored all the point retrivers. The point
retrievers have the particularity to retrieve points from a pool of points.
The retrieve could be auto-retrieve (retrieve the neighbours of some points
from the same pool of points) or cross-retrieve (retrieve points from a
different pool of points).


Compulsary requisits
--------------------
- discretize (function)
- retrieve_neighs_spec (function)
- format_output (function)
- retriever.data (parameters)

TODO
----
- discretize function
- relative_pos function

"""

import numpy as np
from sklearn.neighbors import KDTree

from retrievers import Retriever


###############################################################################
############################## Point Retrievers ###############################
###############################################################################
class PointRetriever(Retriever):
    """Retreiver of points given points and only considering the information
    of the non-retrivable points.
    """
    typeret = 'point'

    def __init__(self, locs, info_ret=None, autolocs=None, pars_ret=None,
                 flag_auto=True, ifdistance=False, info_f=None, tags=None,
                 relative_pos=None):
        "Creation a point retriever class method."
         # Retrieve information
        pars_ret = self.format_pars_ret(pars_ret)
        self.retriever = define_kdretriever(locs, **pars_ret)
        ## Info_ret mangement
        if type(info_ret).__name__ == 'function':
            self.info_f = info_ret
        else:
            self.info_f = info_f
        self.info_ret = self.default_ret_val if info_ret is None else info_ret
        self.ifdistance = ifdistance
        self.relative_pos = relative_pos
        # Location information
        self.locs = None if autolocs is None else autolocs
        self.autolocs = True if self.locs is None else False
        # Filter information
        self.flag_auto = flag_auto

    def discretize(self, i_locs):
        """Format the index retrieving for the proper index of retrieving of
        the type of retrieving.
        """
        if self.check_coord(i_locs):
            if type(i_locs) == list:
                return -1 * np.ones(len(i_locs))
            else:
                return -1
        return i_locs

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def format_pars_ret(self, pars_ret):
        "Format the parameters of retrieval."
        if pars_ret is not None:
            pars_ret = int(pars_ret)
        pars_ret = {'leafsize': pars_ret}
        return pars_ret

    def format_output(self, i_locs, neighs, dists):
        "Format output."
        neighs, dists = self.exclude_auto(i_locs, neighs, dists)
        return neighs, dists


def define_kdretriever(locs, leafsize=None):
    "Define a kdtree for retrieving neighbours."
    leafsize = locs.shape[0]
    leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
    return KDTree(locs, leaf_size=leafsize)


################################ K Neighbours #################################
###############################################################################
class KRetriever(PointRetriever):
    "Class which contains a retriever of K neighbours."
    default_ret_val = 1

    def retrieve_neighs_spec(self, point_i, kneighs_i, ifdistance=False):
        "Function to retrieve neighs in the specific way we want."
        point_i = self.get_loc_i(point_i)
        res = self.retriever.query(point_i, int(kneighs_i), ifdistance)
        if ifdistance:
            res = res[1][0], res[0][0]
        else:
            res = res[0], None
        ## Correct for another relative spatial measure
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever.data)[res[0], :]
            if type(self.relative_pos).__name__ == 'function':
                res = res[0], self.relative_pos(point_i, loc_neighs)
            else:
                res = res[0], self.relative_pos.compute(point_i, loc_neighs)
        return res


################################ R disctance ##################################
###############################################################################
class CircRetriever(PointRetriever):
    "Circular retriever."
    default_ret_val = 0.1

    def retrieve_neighs_spec(self, point_i, radius_i, ifdistance=False):
        "Function to retrieve neighs in the specific way we want."
        point_i = self.get_loc_i(point_i)
        res = self.retriever.query_radius(point_i, radius_i, ifdistance)
        if ifdistance:
            res = res[0][0], res[1][0]
        else:
            res = res[0], None
        ## Correct for another relative spatial measure
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever.data)[res[0], :]
            if type(self.relative_pos).__name__ == 'function':
                res = res[0], self.relative_pos(point_i, loc_neighs)
            else:
                res = res[0], self.relative_pos.compute(point_i, loc_neighs)
        return res
