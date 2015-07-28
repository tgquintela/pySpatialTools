
"""
Tester Recommender
------------------
Module which groups the functions required to test the recommenders.

TODO
----
- format properly

"""

import numpy as np
from sklearn.metrics import confusion_matrix, r2_score


def tester1(idxs, real_idx, success, labels=None):
    "Tester for recommendations using the success binary variable."

    ## 0. Format inputs
    if labels is None:
        labels = np.unique(real_idx)
    success = success.astype(int)

    ## 1. Compute measure
    logi = idxs[:, 0] == real_idx
    r = (success[logi] == 1).sum()/float(success.shape[0])
#    conf_mats = np.zeros((labels.shape[0], labels.shape[0], idxs.shape[1]))
#    for i in range(idxs.shape[1]):
#        conf_mats[:, :, i] = confusion_matrix(idxs[:, i], real_idx)
    return r


def tester2(idxs, real_idx, p_success):
    "Tester for recommendations using probability measure of closing."

    ## 1. Compute measure
    logi = idxs[:, :, 0] == real_idx
    r = (p_success[logi].sum())/float(p_success.shape[0])
    return r


def tester3(Q, success):
    """Use of the auc of the ROC curve for measure how good are the Qs for
    representing the success or not of the event measured by Q.
    """
    ## 0. Format inputs
    closed = closed.astype(int)

    ## 1. Compute measure
    # Compute the measure of ROC curve
    fpr, tpr, thresholds = roc_curve(array_real, array_pred)
    # numerical measure
    measure = auc(fpr, tpr)
    return measure


def tester4(Q, p_success):
    """Tester for the results of the quality measure compared with the
    probability of sucess of this result.
    """
    measure = r2_score(Q, p_success)
    return measure
