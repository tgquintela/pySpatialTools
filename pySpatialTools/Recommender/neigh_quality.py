
"""
Neighbourhood quality
---------------------
Compute the quality from the neighbourhood.

TODO
----
import weighs_creation function
points_arr description
"""

import numpy as np
from pythonUtils.numpy_tools.stats import counting
from pythonUtils.numpy_tools.sorting import get_kbest
from recommender_models import RecommenderModel


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

    def __init__(self, retriever, weights_f):
        """
        Parameters
        ----------
        retriever: pySpatialTools.Retrieve object
            the retriever object to select the neighbours candidates.
        weights_f: function
            a function to compute weighs from distances.
        """
        self.retriever = retriever
        self.weights_f = weights_f

    def compute_quality_measure(self, descrip_matrix, points_arr, feat_arr,
                                val_type=None):
        """Computation of the quality measure associated to the model.

        Parameters
        ----------
        descrip_matrix: np.ndarray, shape (n, m_feats)
            the descriptor matrix of the data in the sample.
        points_arr: np.ndarray, shape (n,)

        feat_arr: numpy.ndarray, shape (n,)
            the categorical features information of each sample we want to
            study.
        val_type: int, np.ndarray or NoneType
            the value type we want to measure its quality.
        retriever: pySpatialTools.Retrieve object
            the information object to retrieve neighbourhood from a location.
        weights_creation: function
            a function which returns an np.ndarray of the weights of each point
            given the distance.

        Returns
        -------
        Q: np.ndarray, shape (n,)
            the quality measure obtained for the sample given.

        """
        Q = compute_quality_measure(descrip_matrix, points_arr, feat_arr,
                                    self.retriever, self.weights_f, val_type)
        return Q

    def compute_kbest_type(self, descrip_matrix, points_arr, feat_arr, kbest):
        """Compute the k best type and their quality.

        Parameters
        ----------
        descrip_matrix: np.ndarray, shape (n, m_feats)
            the descriptor matrix of the data in the sample.
        points_arr: np.ndarray, shape (n,)

        feat_arr: numpy.ndarray, shape (n,)
            the features information of each sample we want to study.
        kbest: int
            the quantity k of best types we want to retrieve with our
            recommendation algorithm.
        retriever: pySpatialTools.Retrieve object
            the information object to retrieve neighbourhood from a location.
        weights_creation: function
            a function which returns an np.ndarray of the weights of each point
            given the distance.

        Returns
        -------
        Qs: np.ndarray, shape (n, kbest)
            the quality measure obtained for the sample.
        idxs: np.ndarray, shape (n, kbest)
            the best types selected by our recommendation algorithm.

        """
        Qs, idxs = compute_kbest_type(descrip_matrix, points_arr, feat_arr,
                                      kbest, self.retriever, self.weights_f)
        return Qs, idxs


def compute_quality_measure(descrip_matrix, points_arr, feat_arr,
                            retriever, weights_creation, val_type=None):
    """Computation of the quality measure associated to the model.

    Parameters
    ----------
    descrip_matrix: np.ndarray, shape (n, m_feats)
        the descriptor matrix of the data in the sample.
    points_arr: np.ndarray, shape (n,)

    feat_arr: numpy.ndarray, shape (n,)
        the categorical features information of each sample we want to study.
    val_type: int, np.ndarray or NoneType
        the value type we want to measure its quality.
    retriever: pySpatialTools.Retrieve object
        the information object to retrieve neighbourhood from a location.
    weights_creation: function
        a function which returns an np.ndarray of the weights of each point
        given the distance.

    Returns
    -------
    Q: np.ndarray, shape (n,)
        the quality measure obtained for the sample given.

    """
    n, n_vals = descrip_matrix.shape
    Q = np.zeros(n)
    for i in xrange(n):
        neighs, dist = retriever.retrieve_neighs(descrip_matrix)
        weights = weights_creation(dist, points_arr[neighs])
        votation = counting(feat_arr[neighs], weights, n_vals)
        if val_type is not None:
            vote = votation[val_type]
        else:
            vote = votation[feat_arr[i, :]]
    return Q


def compute_kbest_type(descrip_matrix, points_arr, feat_arr, kbest, retriever,
                       weights_creation):
    """Compute the k best type and their quality.

    Parameters
    ----------
    descrip_matrix: np.ndarray, shape (n, m_feats)
        the descriptor matrix of the data in the sample.
    points_arr: np.ndarray, shape (n,)

    feat_arr: numpy.ndarray, shape (n,)
        the features information of each sample we want to study.
    kbest: int
        the quantity k of best types we want to retrieve with our
        recommendation algorithm.
    retriever: pySpatialTools.Retrieve object
        the information object to retrieve neighbourhood from a location.
    weights_creation: function
        a function which returns an np.ndarray of the weights of each point
        given the distance.

    Returns
    -------
    Qs: np.ndarray, shape (n, kbest)
        the quality measure obtained for the sample.
    idxs: np.ndarray, shape (n, kbest)
        the best types selected by our recommendation algorithm.

    """
    n, n_vals = descrip_matrix.shape
    Qs, idxs = np.zeros((n, kbest)), np.zeros((n, kbest))
    for i in xrange(descrip_matrix.shape[0]):
        neighs, dist = retriever.retrieve_neighs(descrip_matrix)
        weights = weights_creation(dist, points_arr[neighs])
        votation = counting(feat_arr[neighs], weights, n_vals)
        Qs[i, :], idxs[i, :] = get_kbest(votation, kbest)
    return Qs, idxs
