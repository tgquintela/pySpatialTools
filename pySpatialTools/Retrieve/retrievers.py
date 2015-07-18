
"""
Retrievers
----------
The objects to retrieve neighbours in flat (non-splitted) space.


"""


from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import numpy as np


class Retriever():
    """Class which contains the retriever of points.
    TODO
    ----
    - More information in the defintion of the retriever.
    - Global flag to cross retrieve (not include same points)
    """

    flag_auto = True

    def retrieve_neighs(self, point_i, info_i, ifdistance=False):
        neighs, dist = self.retrieve_neighs_spec(point_i, info_i, ifdistance)
        if not self.flag_auto:
            neighs = self.exclude_auto(point_i, neighs)
        return neighs, dist


###############################################################################
################################# Retrievers ##################################
###############################################################################

################################ K Neighbours #################################
###############################################################################
class KRetriever(Retriever):
    "Class which contains a retriever of K neighbours."

    def __init__(self, locs, flag_auto=True):
        self.retriever = define_kdretriever(locs)

    def retrieve_neighs_spec(self, point_i, info_i, ifdistance=False):
        res = self.retriever.query(point_i, info_i)
        res = res[1], res[0]
        if not ifdistance:
            res = res[1]
        return res

    def exclude_auto(self, point_i, neighs):
        sh = point_i.shape
        point_i = point_i if len(sh) == 2 else point_i.reshape(1, sh[0])
        logi = np.logical_and(self.retriever.data[:, 0] == point_i[:, 0],
                              self.retriever.data[:, 1] == point_i[:, 1])
        to_exclude_points = np.where(logi)[0]
        neighs = [e for e in neighs if e not in to_exclude_points]
        return neighs


################################ R disctance ##################################
###############################################################################
class CircRetriever(Retriever):
    "Circular retriever."

    def __init__(self, locs, flag_auto=True):
        self.retriever = define_kdretriever(locs)
        self.flag_auto = flag_auto

    def retrieve_neighs_spec(self, point_i, info_i, ifdistance=False):
        res = self.retriever.query_ball_point(point_i, info_i)
        if ifdistance:
            if len(point_i.shape) == 1:
                point_i = point_i.reshape(1, point_i.shape[0])
            points = self.retriever.data[res, :]
            if len(points.shape) == 1:
                points = points.reshape(1, points.shape[0])
            dis = cdist(point_i, points)
            res = res, dis
        return res

    def exclude_auto(self, point_i, neighs):
        sh = point_i.shape
        point_i = point_i if len(sh) == 2 else point_i.reshape(1, sh[0])
        logi = np.logical_and(self.retriever.data[:, 0] == point_i[:, 0],
                              self.retriever.data[:, 1] == point_i[:, 1])
        to_exclude_points = np.where(logi)[0]
        neighs = [e for e in neighs if e not in to_exclude_points]
        return neighs


def define_kdretriever(locs):
    leafsize = locs.shape[0]
    leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
    return KDTree(locs, leafsize=leafsize)


############################### Same location #################################
###############################################################################
class SameRegionRetriever(Retriever):
    "Temporal: Problem: not distance between neighs."

    def __init__(self, locs, flag_auto=True):
        self.flag_auto = flag_auto
        self.retriever = define_retriever(locs)

    def retrieve_neighs_spec(self, region_i, info_i, ifdistance=False):
        logi = region_i == regions
        res = list(np.where(logi)[0])
        if distance:
            res = res, np.zeros(len(res))
        return res


def define_retriever(self, regions):
    return regions
