

"""
General pjensen quality
-----------------------
Module which groups all the functions related with the computation of the
quality of a spatial point balancing global stats and correlation with
local stats and the neighbourhood description.

TODO
----
- Different combinations of measures from the base.
"""

import numpy as np
from recommender_models import RecommenderModel

from pythonUtils.numpy_tools.sorting import get_kbest


########### Class for computing index of the model selected
##################################################################
class CorrRecommender(RecommenderModel):
    """Recommender model for location recommendation. This model is an
    extension of the  application of the proposal used by P. Jensen [1]
    It is based on the statical assumption of market, the stationary regime of
    the system, that makes that the average position of a type of point is the
    best location for these type of points.

    TODO
    ----
    """
    name_desc = "Corr recommender"

    def __init__(self):
        "The inputs are the needed to compute model_dim."
        pass

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_quality(self, corr_mat, desc_mat, type_arr, val_type=None):
        "Computation of the quality measure associated to the model."
        Q = compute_quality_measure(corr_mat, desc_mat, type_arr, val_type)
        return Q

    def compute_kbest_type(self, corr_mat, desc_mat, type_arr, kbest):
        "Compute the k best type and their quality."
        Q, idxs = compute_kbest_type(corr_mat, desc_mat, type_arr, kbest)
        return Q, idxs


def compute_quality_measure(corr_mat, desc_mat, type_arr, val_type=None):
    "Main function to compute the quality measure of pjensen."
    ## Compute needed variables
    type_vals = np.unique(type_arr)
    n, n_vals = desc_mat.shape
    ## Loop over each type
    references = compute_references(desc_mat, type_arr, type_vals)

    ## Loop for each row
    Q = np.zeros(n)
    for i in xrange(n):
        reference = select_reference(i, val_type, type_arr, references)
        scaling = select_scaling(i, val_type, type_arr, corr_mat)
        Q[i] = compute_quality_measure(desc_mat[i, :], reference, scaling)
    return Q


def compute_kbest_type(corr_mat, desc_mat, type_arr, kbest):
    "Compute the k best type and their quality."

    ## Compute needed variables
    type_vals = np.unique(type_arr)
    n, n_vals = desc_mat.shape
    ## Loop over each type
    references = compute_references(desc_mat, type_arr, type_vals)

    ## Loop for each row
    Qs = np.zeros((n, kbest))
    idxs = np.zeros((n, kbest)).astype(int)
    for i in xrange(n):
        values = np.zeros(n_vals)
        for k in type_vals:
            reference = select_reference(i, k, type_arr, references)
            scaling = select_scaling(i, k, type_arr, corr_mat)
            values[k] = compute_quality_measure(desc_mat[i, :], reference,
                                                scaling)
        idxs[i], Qs[i] = get_kbest(values, kbest)

    return Q, idxs


### Auxiliar functions
##### ----------------
######################
def compute_quality_measure(description, reference, scaling, method='pjensen'):
    "Auxiliar function to compute measure."
    if type(measure) == str:
        if method == 'pjensen':
            comparison = compute_quality_comparison(description, reference)
            measure = compute_quality_scale(comparison, scaling)
    elif type(measure) in [tuple, list]:
        comparison = compute_quality_comparison(description, reference,
                                                method[0])
        measure = compute_quality_scale(comparison, scaling, method[1])
    elif type(method).__str__ == 'function':
        measure = method(description, reference, scaling)

    return measure


def compute_quality_scale(comparison, scaling, method='dot'):
    """Auxiliar function which carries out with the task to scale the
    comparison between the local description and the global stats.
    """
    if method == 'dot':
        measure = np.dot(scaling, comparison)

    return measure


def compute_quality_comparison(description, reference, method='diff'):
    """Auxiliar function which performs the comparison between the reference
    descriptors and the actual local descriptors.
    """
    if method == 'diff':
        comparison = description - reference
    elif method == 'absdiff':
        comparison = np.abs(description - reference)

    return comparison


def select_reference(i, val_type, type_arr, references):
    "Select the reference from the references."
    if val_type is not None:
        val_j = val_type
    else:
        val_j = type_arr[i, 0]
    reference = references[val_j, :]
    return reference


def select_scaling(i, val_type, type_arr, corr_mat, scale_f=None):
    "Select the reference from the scaling."
    if val_type is not None:
        val_j = val_type
    else:
        val_j = type_arr[i, 0]
    scaling = corr_mat[val_j, :]
    return scaling


def compute_references(desc_mat, type_arr, type_vals, ref_f=None):
    "Function to compute the references, using global stats."
    n_vals, m_vals = type_vals.shape[0], desc_mat.shape[1]
    type_arr = type_arr.ravel().astype(int)
    refs = np.zeros((n_vals, m_vals))
    for k in type_vals:
        refs[k, :] = np.mean(desc_mat[type_arr == k, :], axis=0)
    return refs
