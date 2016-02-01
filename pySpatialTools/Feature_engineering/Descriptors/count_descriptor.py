
"""
Count descriptors
-----------------
Module which groups the methods related with computing histogram-based spatial
descriptors.


Parameters
----------
out

"""

from collections import Counter
import numpy as np
from pySpatialTools.Feature_engineering.descriptormodel import DescriptorModel

from pySpatialTools.IO.general_mapper import Map_Vals_i


class Countdescriptor(DescriptorModel):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in feat_arr.

    Parameters
    ----------
    features: numpy.ndarray, shape (n, 2)
        the element-features of the system.
    sp_typemodel: str, object, ...
        the information of the type global output return.
    ####################################################### to featureretreiver
    mapper: dict
        map the type of feature to a number that could act as an index.

    """
    name_desc = "Counting descriptor"
    _n = 0
    _nullvalue = 0

    def __init__(self, features, sp_typemodel='matrix'):
        "The inputs are the needed to compute model_dim."
        ## Format features
        self._format_features(features)
        ## Type of built result
        self._format_map_vals_i(sp_typemodel)
        ## Format function to external interaction and building results
        self._format_result_building()

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_characs(self, pointfeats, point_pos):
        "Compulsary function to pass for the feture retriever."
        pointfeats = np.array(pointfeats).astype(int)
        descriptors = dict(Counter(pointfeats.ravel()))
        ## TODO: Transform dict to array and reverse
        #keys = [self.mapper[key] for key in counts.keys()]
        #descriptors[0, keys] = counts.values()
        return descriptors

    def reducer(self, aggdescriptors_idxs, point_aggpos):
        """Reducer gets the aggdescriptors of the neighbourhood regions
        aggregated and collapse all of them to compute the descriptor
        associated to a retrieved neighbourhood.
        """
        ## 0. To array
        if type(aggdescriptors_idxs) == list:
            if type(aggdescriptors_idxs[0]) == np.ndarray:
                aggdescriptors_idxs = np.array(aggdescriptors_idxs)
        ## 1. Counts array and dict
        if type(aggdescriptors_idxs) == np.ndarray:
            descriptors = np.sum(aggdescriptors_idxs, axis=0)
        elif type(aggdescriptors_idxs) == list:
            if type(aggdescriptors_idxs[0]) == dict:
                vars_ = []
                for i in xrange(aggdescriptors_idxs):
                    vars_.append(aggdescriptors_idxs[i].keys())
                vars_ = set(vars_)
                descriptors = {}
                for e in vars_:
                    descriptors[e] = 0
                    for i in xrange(aggdescriptors_idxs):
                        if e in aggdescriptors_idxs[i].keys():
                            descriptors[e] += aggdescriptors_idxs[i][e]
        return descriptors

    def compute_aggcharacs_i(self, neighs_i, dists_i):
        """Compute aggregated characters for the region i from neighs_i points
        and relative position of neighbourhood points dists_i.

        Parameters
        ----------
        neighs_i: numpy.ndarray
            the points which conforms the neighbourhood.
        dists_i: numpy.ndarray
            the relative position respect the original region.

        Returns
        -------
        aggcharacs_i: numpy.ndarray
            the information aggregated information features.

        """
        counts = Counter(self.features[list(neighs_i), :].ravel())
        aggcharacs_i = self.initialization_desc()
        keys = [self.mapper[key] for key in counts.keys()]
        aggcharacs_i[0, keys] = counts.values()
        return aggcharacs_i

    ###########################################################################
    ########################## Auxiliary functions ############################
    ###########################################################################
    def _compute_featuresnames(self, featureobject):
        "Compute the possible feature names from the pointfeatures."
        if type(featureobject) == np.ndarray:
            featuresnames = list(np.unique(featureobject[:, 0]))
            return featuresnames
        if featureobject._type == 'aggregated':
            featuresnames = featureobject.variables
        elif featureobject._type == 'point':
            if featureobject.variables is None:
                featuresnames = list(np.unique(featureobject.features[:, 0]))
            else:
                featuresnames = featureobject.variables
        return featuresnames

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################
    def _format_map_vals_i(self, sp_typemodel):
        "Format mapper to indicate external val_i to aggregate result."
        ## Preparing input
        n_in, n_out = None, None
        if type(sp_typemodel) == tuple:
            if len(sp_typemodel) == 2:
                n_out = sp_typemodel[1]
            elif len(sp_typemodel) == 3:
                n_in = sp_typemodel[1]
                n_out = sp_typemodel[2]
            sp_typemodel = sp_typemodel[0]
        ## Preparing mapper
        if type(sp_typemodel) == str:
            if sp_typemodel == 'correlation':
                array = self.features[0].features.features[:, 0].astype(int)
                self._map_vals_i = Map_Vals_i(array)
            elif sp_typemodel == 'matrix':
                funct = lambda idx: idx
                self._map_vals_i = Map_Vals_i(funct)
        elif type(sp_typemodel) == np.ndarray:
            self._map_vals_i = Map_Vals_i(sp_typemodel)
        elif type(sp_typemodel).__name__ in ['instance', 'function']:
            self._map_vals_i = Map_Vals_i(sp_typemodel)
        self._map_vals_i.n_in = n_in
        self._map_vals_i.n_out = n_out

    def _format_result_building(self):
        "Format how to build and aggregate results."
        ## Size of the possible results.
        n_vals_i = self._map_vals_i.n_out
        n_feats = self.features.nfeats
        ## Initialization features
        self.initialization_desc = lambda: np.zeros((1, n_feats))
        ## Global construction of result
        if n_vals_i is not None:
            n_pert = self.features.k_perturb + 1
            ## Init global result
            self.initialization_output =\
                lambda: np.zeros((n_vals_i, n_feats, n_pert))
            ## Adding result
            self.add2result = lambda x, x_i: x + x_i
        else:
            ##### WARNING: Problem not solved
            ## Init global result
            self.initialization_output = lambda: []
            ## Adding result
            self.add2result = lambda x, x_i: x.append(x_i)

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return corr_loc
