

"""
Pjensen descriptors
-------------------
Module which groups all the functions related with the computation of the
spatial correlation using Jensen model.

TODO
----
- Support for more than 1 dimensional type_var.
"""

import numpy as np
from recommender_models import RecommenderModel

from pythonUtils.sorting import get_kbest


########### Class for computing index of the model selected
##################################################################
class Pjensen(RecommenderModel):
    """
    Recommender model for location recommendation. This model is the
    application of the proposal used by P. Jensen [1]
    It is based on the statical assumption of market, the stationary regime of
    the system, that makes that the average position of a type of point is the
    best location for these type of points.

    References
    ----------
    .. [1]

    TODO
    ----

    """
    name_desc = "PJensen recommender"

    def __init__(self, df, typevars):
        "The inputs are the needed to compute model_dim."
        self.typevars = typevars
        self.counts, self.counts_info = compute_globalstats(df, typevars)
        self.n_vals = self.counts_info.shape[0]
        self.model_dim = self.compute_model_dim()
        self.globalnorm = self.compute_global_info_descriptor(df.shape[0])

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_quality(self, corr_loc, count_matrix, feat_arr, val_type=None):
        "Computation of the quality measure associated to the model."
        Q = compute_quality_measure(corr_loc, count_matrix, feat_arr, val_type)
        return Q


def compute_quality_measure(corr_loc, count_matrix, feat_arr, val_type=None):
    "Main function to compute the quality measure of pjensen."
    ## Compute needed variables
    type_vals = np.unique(feat_arr)
    n, n_vals = count_matrix.shape
    ## Loop over each type
    avges = np.zeros((n_vals, n_vals))
    for val_j in type_vals:
        avges[val_j, :] = np.mean(count_matrix[feat_arr.ravel() == val_j, :],
                                  axis=0)
    ## Loop for each
    Q = np.zeros(n)
    for i in xrange(n):
        if val_type is not None:
            val_j = val_type
        else:
            val_j = feat_arr[i, 0]
        avg = avges[val_j, :]
        Q[i] = np.sum(corr_loc[val_j, :] * (count_matrix[i, :] - avg))
    return Q


def jensen_quality(val_j, count_matrix, corr_loc, avg):
    ""
    Q = np.sum(corr_loc[val_j, :] * (count_matrix[i, :] - avg))

    Q = np.mean(count_matrix[feat_arr.ravel() == val_j, :], axis=0)
    return Q


def compute_kbest_type(corr_loc, count_matrix, feat_arr, kbest):
    "Compute the k best type and their quality."

    ## Compute needed variables
    type_vals = np.unique(feat_arr)
    n, n_vals = count_matrix.shape
    ## Loop over each type
    avges = np.zeros((n_vals, n_vals))
    for val_j in type_vals:
        avges[val_j, :] = np.mean(count_matrix[feat_arr.ravel() == val_j, :],
                                  axis=0)
    ## Loop for each
    Qs = np.zeros((n, kbest))
    idxs = np.zeros((n, kbest)).astype(int)
    for i in xrange(n):
        values = np.zeros(n_vals)
        for k in type_vals:
            avg = avges[k, :]
            values[k] = np.sum(corr_loc[k, :] * (count_matrix[i, :] - avg))
        idxs[i], Qs[i] = get_kbest(values, kbest)

    return Q, idxs

