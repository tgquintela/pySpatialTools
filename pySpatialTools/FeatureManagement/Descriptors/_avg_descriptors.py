
"""
Avg descriptors
---------------
Module which groups the methods related with computing average-based spatial
descriptors.

"""

import numpy as np
from pySpatialTools.Feature_engineering.descriptormodel import DescriptorModel


class AvgDescriptor(DescriptorModel):
    """Model of spatial descriptor computing by averaging the type of the
    neighs represented in feat_arr.

    Parameters
    ----------
    features: numpy.ndarray, shape (n, 2)
        the element-features of the system.
    sp_typemodel: str, object, ...
        the information of the type global output return.

    """
    name_desc = "Average descriptor"

    def __init__(self, feat_arr, sp_typemodel='matrix'):
        "The inputs are the needed to compute model_dim."
        self.features = feat_arr
        ## Type of built result
        self.sp_typemodel = sp_typemodel
        ## Compute initialization descriptors
        n_feats = feat_arr.shape[1]
        nvals_i = self.compute_nvals_i()
        self.initialization_desc = lambda: np.zeros((1, n_feats))
        self.initialization_output = lambda x: np.zeros((nvals_i, n_feats, x))
        ## Adding result
        self.add2result = lambda x, x_i: x + x_i

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_predescriptors(self, i, neighs, dists, reindices, k):
        """Compute descriptors from the i, neigh and distances values regarding
        the permutation information.

        Parameters
        ----------
        i: int
            the indice of the element we want to compute the spatial
            descriptors using their relative position with its environment
            represented as their neighs and the dists information.
        neighs: list or numpy.ndarray
            the information of the neighbourhood elements of i.
        dists: numpy.ndarray
            a measure of the relative position between i and its neighs.
        reindices: numpy.ndarray
            the reindices matrix of a permutation.
        k: int
            the index of permutation used.

        Returns
        -------
        descriptors: numpy.ndarray
            the information descriptors.

        """
        ## Compute needed values
        i, neighs = reindices[i, :], reindices[neighs, :]
        ## Compute descriptors
        descriptors = self.initialization_desc()
        descriptors[0, :] = np.mean(self.features[neighs, :], axis=0)
        return descriptors

    def compute_value_i(self, i, k, reindices):
        "Compute the val of a specific point."
        if type(self.sp_typemodel) == str:
            if self.sp_typemodel == 'correlation':
                val_i = self.features[reindices[i, k], :].astype(int)
            elif self.sp_typemodel == 'matrix':
                val_i = reindices[i, k]
        else:
            self.sp_typemodel(self, i, k, reindices)
        return val_i

    def compute_nvals_i(self):
        "Compute the val of a specific point."
        if type(self.sp_typemodel) == str:
            if self.sp_typemodel == 'correlation':
                nvals_i = 1
            elif self.sp_typemodel == 'matrix':
                nvals_i = self.features.shape[0]
        else:
            nvals_i = self.sp_typemodel.get_nvals_i(self)
        return nvals_i

    ###########################################################################
    ####################### Non-compulsary functions ##########################
    ###########################################################################
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
        aggcharacs_i = self.initialization_desc()
        aggcharacs_i = np.mean(self.features[list(neighs_i), :], axis=0)
        return aggcharacs_i

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return corr_loc
