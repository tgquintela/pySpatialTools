
"""
Tester Recommender
------------------
Module which groups the functions required to test the recommenders.
There are different types of recommender test measures.

TODO
----
- format properly

"""

import numpy as np
from sklearn.metrics import confusion_matrix, r2_score


def binary_categorical_measure(idxs, real_idx, success):
    """Tester for recommendations using the success binary variable. It
    computes the ratio of success for the properly predicted idxs.

    Parameters
    ----------
    idxs: np.ndarray of ints
        the predicted type recomendation.
    real_idx: np.ndarray of ints
        the real type.
    success: np.ndarray of {0, 1}
        the array which represents a categorical measure of success. It is like
        a real quality measure we want to predict.

    Returns
    -------
    r: float
        the measure of performance.

    """

    ## 0. Format inputs
    if labels is None:
        labels = np.unique(real_idx)
    success = success.astype(int)
    idxs = idxs.reshape(idxs.shape[0], 1) if len(idxs.shape) == 1 else idxs

    ## 1. Compute measure
    logi = idxs[:, 0] == real_idx
    r = (success[logi] == 1).sum()/float(success.shape[0])
#    conf_mats = np.zeros((labels.shape[0], labels.shape[0], idxs.shape[1]))
#    for i in range(idxs.shape[1]):
#        conf_mats[:, :, i] = confusion_matrix(idxs[:, i], real_idx)
    return r


def float_categorical_measure(idxs, real_idx, p_success):
    """Tester for recommendations using probability measure of closing.

    Parameters
    ----------
    idxs: np.ndarray of ints
        the predicted type recomendation.
    real_idx: np.ndarray of ints
        the real type.
    p_success: np.ndarray
        the array of success points. It is a measure of success. It is like a
        real quality measure we want to predict.

    Returns
    -------
    r: float

    """

    ## 1. Compute measure
    logi = idxs[:, :, 0] == real_idx
    r = (p_success[logi].sum())/float(p_success.shape[0])
    return r


def binary_quality_measure(Q, success):
    """Use of the auc of the ROC curve for measure how good are the Qs for
    representing the success or not of the event measured by Q.

    Parameters
    ----------
    Q: np.ndarray, shape (n,)
        the quality measure obtained by the recommender.
    success: np.ndarray of ints, shape (n,)
        the 0,1-measure of success. It is as a real quality measure we want to
        predict.

    Returns
    -------
    measure: float
        measure of performance of the recommendation.

    """
    ## 0. Format inputs
    closed = closed.astype(int)

    ## 1. Compute measure
    # Compute the measure of ROC curve
    fpr, tpr, thresholds = roc_curve(array_real, array_pred)
    # numerical measure
    measure = auc(fpr, tpr)
    return measure


def float_quality_measure(Q, p_success):
    """Tester for the results of the quality measure compared with the
    probability of sucess of this result.

    Parameters
    ----------
    Q: np.ndarray, shape (n,)
        the quality vector.
    p_success: np.ndarray, shape (n,)
        the measure of success of the location. It is like the real quality
        measure.

    Returns
    -------
    measure: float
        measure of performance.

    """
    measure = r2_score(Q, p_success)
    return measure
