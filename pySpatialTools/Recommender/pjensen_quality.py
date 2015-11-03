
"""
Pjensen quality
---------------
Module which groups all the functions related with the computation of the
Pjensen quality.

TODO
----
Check correct inputs for variables.
More than 1-dim feat_arr. TO generalize.


Structure
---------
class Pjensen
 |
 |- functions compute_quality_measure, compute_kbest_type
    |
    |- function compute_avges_by_val

"""

import numpy as np
from recommender_models import RecommenderModel

from pythonUtils.numpy_tools.sorting import get_kbest


########### Class for computing index of the model selected
##################################################################
class PjensenRecommender(RecommenderModel):
    """Recommender model for location recommendation. This model is the
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

    def __init__(self):
        "The inputs are the needed to compute model_dim."
        pass

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_quality(self, corr_loc, count_matrix, feat_arr, val_type=None):
        """Computation of the quality measure associated to the model.

        Parameters
        ----------
        corr_loc: numpy.ndarray, shape (Nvars, Nvars)
            the correlation matrix between the whole set of variables.
        count_matrix: numpy.ndarray, shape (N, Nvars)
            the count matrix of each type of variable.
        feat_arr: numpy.ndarray, shape (N, Nfeats)
            the features information of each sample we want to study.
        val_type: int or numpy.ndarray
            the type of element we want to measure its quality.

        Returns
        -------
        Q: numpy.ndarray, shape (N,)
            the quality values for each sample.

        """
        Q = compute_quality_measure(corr_loc, count_matrix, feat_arr, val_type)
        return Q

    def compute_kbest_type(self, corr_loc, count_matrix, feat_arr, kbest):
        """Compute the k best type and their quality.

        Parameters
        ----------
        corr_loc: numpy.ndarray, shape (Nvars, Nvars)
            the correlation matrix between the whole set of variables.
        count_matrix: numpy.ndarray, shape (N, Nvars)
            the count matrix of each type of variable.
        feat_arr: numpy.ndarray, shape (N, Nfeats)
            the features information of each sample we want to study.
        kbest: int
            the number of best types we want to get.

        Returns
        -------
        Qs: numpy.ndarray, shape (N, kbest)
            the quality values for each sample.
        idxs: numpy.ndarray, shape (N, kbest)
            the indices of the k-best types for each sample.

        """
        Q, idxs = compute_kbest_type(corr_loc, count_matrix, feat_arr, kbest)
        return Q, idxs


def compute_quality_measure(corr_loc, count_matrix, feat_arr, val_type=None):
    """Main function to compute the quality measure of pjensen.

    Parameters
    ----------
    corr_loc: numpy.ndarray, shape (Nvars, Nvars)
        the correlation matrix between the whole set of variables.
    count_matrix: numpy.ndarray, shape (N, Nvars)
        the count matrix of each type of variable.
    feat_arr: numpy.ndarray, shape (N, Nfeats)
        the features information of each sample we want to study.
    val_type: int or numpy.ndarray
        the type of element we want to measure its quality.

    Returns
    -------
    Q: numpy.ndarray, shape (N,)
        the quality values for each sample.

    """
    ## Compute needed variables
    type_vals = np.unique(feat_arr)
    n, n_vals = count_matrix.shape
    ## Loop over each type
    avges = compute_avges_by_val(count_matrix, feat_arr, type_vals)
    ## Loop for each row
    Q = np.zeros(n)
    for i in xrange(n):
        if val_type is not None:
            val_j = val_type
        else:
            val_j = feat_arr[i, 0]
        avg = avges[val_j, :]
        Q[i] = np.sum(corr_loc[val_j, :] * (count_matrix[i, :] - avg))
    return Q


def compute_kbest_type(corr_loc, count_matrix, feat_arr, kbest):
    """Compute the k best type and their quality.

    Parameters
    ----------
    corr_loc: numpy.ndarray, shape (Nvars, Nvars)
        the correlation matrix between the whole set of variables.
    count_matrix: numpy.ndarray, shape (N, Nvars)
        the count matrix of each type of variable.
    feat_arr: numpy.ndarray, shape (N, Nfeats)
        the features information of each sample we want to study.
    kbest: int
        the number of best types we want to get.

    Returns
    -------
    Qs: numpy.ndarray, shape (N, kbest)
        the quality values for each sample.
    idxs: numpy.ndarray, shape (N, kbest)
        the indices of the k-best types for each sample.

    """
    ## Compute needed variables
    type_vals = np.unique(feat_arr)
    n, n_vals = count_matrix.shape
    ## Loop over each type
    avges = compute_avges_by_val(count_matrix, feat_arr, type_vals)
    ## Loop for each row
    Qs = np.zeros((n, kbest))
    idxs = np.zeros((n, kbest)).astype(int)
    for i in xrange(n):
        values = np.zeros(n_vals)
        for k in type_vals:
            avg = avges[k, :]
            values[k] = np.sum(corr_loc[k, :] * (count_matrix[i, :] - avg))
        idxs[i], Qs[i] = get_kbest(values, kbest)

    return Qs, idxs


def compute_avges_by_val(count_matrix, feat_arr, type_vals):
    """Compute the average for each type value.

    Parameters
    ----------
    count_matrix: numpy.ndarray, shape (N, Nvars)
        the count matrix of each type of variable.
    feat_arr: numpy.ndarray, shape (N, Nfeats)
        the features information of each sample we want to study.
    type_vals: numpy.ndarray, shape (Nvals,)
        the differet vals we can have in the feat_arr.

    Returns
    -------
    avges: numpy.ndarray, shape (Nvals, Nvals)
        the averages by value.

    See also
    --------
    pjensen_quality.compute_kbest_type, pjensen_quality.compute_quality_measure

    """
    n_vals = type_vals.shape[0]
    avges = np.zeros((n_vals, n_vals))
    for val_j in type_vals:
        avges[val_j, :] = np.mean(count_matrix[feat_arr.ravel() == val_j, :],
                                  axis=0)
    return avges
