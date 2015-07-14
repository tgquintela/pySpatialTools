
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
    More information in the defintion of the retriever.
    """

    def __init__(self, locs):
        self.retriever = self.define_retriever(locs)

    def retrieve_neighs(self, point_i, info_i, ifdistance=False):
        self.retrieve_neighs_spec(point_i, info_i, ifdistance)


###############################################################################
################################# Retrievers ##################################
###############################################################################
class KRetriever(Retriever):
    "Class which contains a retriever of K neighbours."

    def define_retriever(self, locs):
        leafsize = locs.shape[0]
        leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        return KDTree(locs, leafsize=leafsize)

    def retrieve_neighs_spec(self, point_i, info_i, ifdistance=False):
        res = self.retriever.query(point_i, info_i)
        if not ifdistance:
            res = res[1]
        return res


class CircRetriever(Retriever):
    "Circular retriever."
    def define_retriever(self, locs):
        leafsize = locs.shape[0]
        leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        return KDTree(locs, leafsize=leafsize)

    def retrieve_neighs_spec(self, point_i, info_i, ifdistance=False):
        res = self.retriever.query_ball_point(point_i, info_i)
        if ifdistance:
            aux = cdist(point_i, self.retriever.data[res, :])
            res = aux, res
        return res

class SameRegionRetriever(Retriever):
    "Temporal: Problem: not distance between neighs."
    def define_retriever(self, regions):
        return regions

    def retrieve_neighs_spec(self, region_i, info_i, ifdistance=False):
        logi = region_i == regions
        res = list(np.where(logi)[0])
        if distance:
            res = res, np.zeros(len(res))
        return res
