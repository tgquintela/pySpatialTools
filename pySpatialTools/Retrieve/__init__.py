
"""
Module oriented to compute neighbourhood.
"""


from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import numpy as np

from Mscthesis.Preprocess import Aggregator


class Neighbourhood():
    """
    Retrieve neighs.
    """
    aggdiscretizor = None
    aggretriever = None
    aggfeatures = None
    agglocs = None

    retriever = None

    cond_funct = lambda xself, xbool: xbool

    def __init__(self, retriever, typevars, df, reindices, aggretriever,
                 funct=None, discretizor=None):
        self.define_mainretriever(retriever)
        self.define_aggretriever(typevars, df, reindices, aggretriever, funct,
                                 discretizor)

    def define_mainretriever(self, retriever):
        self.retriever = retriever

    def define_aggretriever(self, typevars, df, reindices, aggretriever,
                            funct=None, discretizor=None):
        """Main function to define an aggretriever. It will save the needed
        information in order to retrieve the neighbours aggregate regions.
        """
        self.aggdiscretizor, self.aggretriever = discretizor, aggretriever
        if discretizor is not None:
            locs = df[typevars['loc_vars']].as_matrix()
            agg_arr = discretizor.map2id(locs)
            agg_arr = pd.DataFrame(agg_arr, columns=['agg_var'])
            ## Ouputs
            df = pd.concat([df, agg_arr], axis=1)
            typevars['agg_var'] = 'agg_var'
        ## Creation of the aggregator object and aggregation
        agg = Aggregator(typevars)
        agglocs, aggfeatures = agg.retrieve_aggregation(df, reindices, funct)
        ## Correction of the agglocs if there is a discretizor
        if discretizor is not None:
            agglocs = discretizor.map2aggloc(agglocs)
        self.aggfeatures, self.agglocs = aggfeatures, agglocs

    def retrieve_neighs(self, point_i, cond_i, info_i):
        """Retrieve the neighs information and the type of retrieving.
        Type of retrieving:
        - aggfeatures: aggregate
        - indices of neighs: neighs_i
        """
        if len(point_i.shape) == 1:
            point_i = point_i.reshape(1, point_i.shape[0])
        typereturn = self.get_type_return(cond_i)
        if typereturn:
            neighbourhood = self.retrieve_neighs_agg(point_i, info_i)
        else:
            neighbourhood = self.retrieve_neighs_i(point_i, info_i)
        typereturn = 'aggregate' if typereturn else 'individual'
        return neighbourhood, typereturn

    def retrieve_neighs_agg(self, point_i, info_i):
        "Retrieve the correspondent regions."
        ## Discretizor

        if type(self.aggretriever) != np.ndarray:
            out = self.aggretriever.map2id(point_i)
        else:
            out = self.aggretriever
        return out

    def retrieve_neighs_i(self, point_i, info_i):
        "Retrieve the neighs."
        return self.retriever.retrieve_neighs(point_i, info_i)

    ###########################################################################
    ########################## Condition aggregation ##########################
    ###########################################################################
    def set_aggcondition(self, f):
        "Setting condition function for aggregate data retrieval."
        self.cond_funct = f

    def get_type_return(self, cond_i):
        "Apply condition setted."
        ## TODO: Add the possibility to not be in aggregate and return False
        return self.cond_funct(cond_i)


###############################################################################
################################# Retrievers ##################################
###############################################################################
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
