
"""
Point retriever
---------------
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
                 flag_auto=True, ifdistance=False, info_f=None):
        "Creation a point retriever class method."
         # Retrieve information
        pars_ret = self.format_pars_ret(pars_ret)
        self.retriever = define_kdretriever(locs, **pars_ret)
        self.info_ret = self.default_ret_val if info_ret is None else info_ret
        self.info_f = info_f
        self.ifdistance = ifdistance
        # Location information
        self.autolocs = True if autolocs is None else False
        self.locs = None if autolocs is None else autolocs
        # Filter information
        self.flag_auto = flag_auto

    def discretize(self, i_locs):
        """Format the index retrieving for the proper index of retrieving of
        the type of retrieving.
        """
        if self.check_coord:
            return -1 * np.ones(i_locs.shape[0])
        return i_locs

    def format_pars_ret(self, pars_ret):
        "Format the paramters of retrieval."
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
        return res
