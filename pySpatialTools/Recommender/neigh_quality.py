
"""
Neighbourhood quality
---------------------
Compute the quality from the neighbourhood.

TODO
----

"""

import numpy as np
from pythonUtils.numpy_tools.stats import counting
from pythonUtils.numpy_tools.sorting import get_kbest


class NeighRecommender(RecommenderModel):
    """Recommender model for location recommendation.
    It is based on the statical assumption of market, the stationary regime of
    the system, that makes that the average position of a type of point is the
    best location for these type of points. In order to obtain the best ones we
    will search the most similar in the descriptor space of the local
    neighbourhood.

    TODO
    ----
    """
    name_desc = "Neighbourhood recommender"

    def __init__(self, retriever):
        self.retriever = retriever

    def compute_quality_measure(self, descrip_matrix, points_arr, feat_arr,
                                val_type=None):
        "Computation of the quality measure associated to the model."
        Q = compute_quality_measure(descrip_matrix, points_arr, feat_arr,
                                    val_type)
        return Q

    def compute_kbest_type(self, descrip_matrix, points_arr, feat_arr, kbest):
        "Compute the k best type and their quality."
        Q, idxs = compute_kbest_type(descrip_matrix, points_arr, feat_arr,
                                     kbest)
        return Q, idxs


def compute_quality_measure(descrip_matrix, points_arr, feat_arr,
                            val_type=None):
    """Computation of the quality measure associated to the model.

    Parameters
    ----------

    Returns
    -------


    """

    n, n_vals = descrip_matrix.shape
    Q = np.zeros(n)
    for i in xrange(n):
        neighs, dist = retriever.retrieve_neighs(descrip_matrix)
        weights = weights_creation(dist, points_arr[neighs])
        votation = counting(feat_arr[neighs], weights, n_vals)
        if val_type is None:
            vote = votation[val_type]
        else:
            vote = votation[feat_arr[i, :]]
    return Q


def compute_kbest_type(descrip_matrix, points_arr, feat_arr, kbest):
    """Compute the k best type and their quality.

    Parameters
    ----------


    Returns
    -------

    """
    n, n_vals = descrip_matrix.shape
    votes, idxs = np.zeros((n, kbest)), np.zeros((n, kbest))
    for i in xrange(descrip_matrix.shape[0]):
        neighs, dist = retriever.retrieve_neighs(descrip_matrix)
        weights = weights_creation(dist, points_arr[neighs])
        votation = counting(feat_arr[neighs], weights, n_vals)
        votes[i, :], idxs[i, :] = get_kbest(votation, kbest)
    return Q, idxs
