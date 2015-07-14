
"""
Neighbourhood definition module
-------------------------------
This module contains the class which performs the neighbourhood retrieval from
spatial data regarding possible aggregation.

"""

import numpy as np
import pandas as pd

from pySpatialTools.Preprocess import Aggregator


class Neighbourhood():
    """
    Retrieve neighs. It is the main class for retrieve neighbourhood. It could
    retrieve the nearer neighbours in a flat space or the nearer neighbourhoods
    in the pre-discretized space.
    """
    aggdiscretizor = None  # discretize space
    aggretriever = None  # aggregation retriever, to retrieve aggregation neigh
    aggfeatures = None  # aggregation features.
    agglocs = None  # locations of the aggregations.

    retriever = None  # main retriever for point2point

    cond_funct = lambda xself, xbool: xbool

    def __init__(self, retriever, typevars, df, reindices, aggretriever,
                 funct=None, discretizor=None):
        self.define_mainretriever(retriever)
        self.define_aggretriever(typevars, df, reindices, aggretriever, funct,
                                 discretizor)

    def define_mainretriever(self, retriever):
        """Main retriever or directed retriever definition. No aggregation
        considered.

        Parameters
        ----------
        retriever: pySpatialTools.retriever class
            the retriever class.

        """
        self.retriever = retriever

    def define_aggretriever(self, typevars, df, reindices, aggretriever,
                            funct=None, discretizor=None):
        """Main function to define an aggretriever. It will save the needed
        information in order to retrieve the neighbours aggregate regions.

        Parameters
        ----------
        typevars: dict
            dictionary with the variables types (loc_vars, feat_vars)
        df: pd.DataFrame
            the data to study.
        reindices: array_like
            the information of permutations.
        aggretriever: retriever object
            the object to retrieve the nearer neighbourhoods.
        funct: function
            function to aggregate 
        discretizor:

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
        """Apply condition setted for deciding if neighbourhood of point i is
        retrieved aggregated or not.

        Parameters
        ----------
        cond_i: arbitrary
            information for decide retrieval aggregated or not.

        Returns
        -------
        output: boolean
            retrieve aggregated or not.

        """
        ## TODO: Add the possibility to not be in aggregate and return False
        return self.cond_funct(cond_i)

