
"""
Reducers
--------
Module which contains different useful reducer functions in order to transform
neighbourhood aggregated features to neighbourhood descriptors.
Reducer gets the aggdescriptors of the neighbourhood regions aggregated and
collapse all of them to compute the descriptor associated to a retrieved
neighbourhood.

This function is a compulsary function in the descriptor model object in
order to be passed to the feture retriever.

** Main properties:
   ---------------
INPUTS:
- aggdescriptors_idxs: the features associated directly to each aggregation.
    The could be expressed in list, np.ndarray or dict.
- point_aggpos: relative position of the aggregations to the element
    neighbourhood.

OUTPUTS:
- descriptors: in dict or array format depending on the descriptors convenience

Standards
---------
- aggdescriptors_idxs: [iss][nei][nfeats]
- point_aggpos: [iss][nei][dim]

"""

import numpy as np
from collections import Counter


def sum_reducer(aggdescriptors_idxs, point_aggpos):
    """This reducer sum all possible aggregation features.

    Parameters
    ----------
    aggdescriptors_idxs: list, np.ndarray or dict [iss][nei][nfeats]
        The features associated directly to each aggregation.
    point_aggpos: optional [iss][nei][dim]
        relative position of the aggregations to the element neighbourhood.

    Returns
    -------
    descriptors: dict, np.ndarray
        The computed descriptors.

    """
    descriptors = []
    for iss in range(len(aggdescriptors_idxs)):
        aux = sum_reducer_ind(aggdescriptors_idxs[iss], point_aggpos[iss])
        descriptors.append(aux)
    return descriptors


def avg_reducer(aggdescriptors_idxs, point_aggpos):
    """This reducer average all possible aggregation features.

    Parameters
    ----------
    aggdescriptors_idxs: list, np.ndarray or dict [iss][nei][nfeats]
        The features associated directly to each aggregation.
    point_aggpos: optional [iss][nei][dim]
        relative position of the aggregations to the element neighbourhood.

    Returns
    -------
    descriptors: dict, np.ndarray
        The computed descriptors.

    """
    descriptors = []
    for iss in range(len(aggdescriptors_idxs)):
        aux = avg_reducer_ind(aggdescriptors_idxs[iss], point_aggpos[iss])
        descriptors.append(aux)
    return descriptors


def sum_reducer_ind(aggdescriptors_idxs, point_aggpos):
    """This reducer sum all possible aggregation features.

    Parameters
    ----------
    aggdescriptors_idxs: list, np.ndarray or dict [iss][nei][nfeats]
        The features associated directly to each aggregation.
    point_aggpos: optional [iss][nei][dim]
        relative position of the aggregations to the element neighbourhood.

    Returns
    -------
    descriptors: dict, np.ndarray
        The computed descriptors.

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
            for i in xrange(len(aggdescriptors_idxs)):
                vars_ += aggdescriptors_idxs[i].keys()
            vars_ = set(vars_)
            descriptors = {}
            for e in vars_:
                descriptors[e] = 0
                for i in xrange(len(aggdescriptors_idxs)):
                    if e in aggdescriptors_idxs[i].keys():
                        descriptors[e] += aggdescriptors_idxs[i][e]
    return descriptors


def avg_reducer_ind(aggdescriptors_idxs, point_aggpos):
    """This reducer average all possible aggregation features.

    Parameters
    ----------
    aggdescriptors_idxs: list, np.ndarray or dict [iss][nei][nfeats]
        The features associated directly to each aggregation.
    point_aggpos: optional [iss][nei][dim]
        relative position of the aggregations to the element neighbourhood.

    Returns
    -------
    descriptors: dict, np.ndarray
        The computed descriptors.

    """
    ## 0. To array
    if type(aggdescriptors_idxs) == list:
        if type(aggdescriptors_idxs[0]) == np.ndarray:
            aggdescriptors_idxs = np.array(aggdescriptors_idxs)
    ## 1. Counts array and dict
    if type(aggdescriptors_idxs) == np.ndarray:
        descriptors = np.mean(aggdescriptors_idxs, axis=0)
    elif type(aggdescriptors_idxs) == list:
        if type(aggdescriptors_idxs[0]) == dict:
            vars_ = []
            for i in xrange(len(aggdescriptors_idxs)):
                vars_ += aggdescriptors_idxs[i].keys()
            count_vars = Counter(vars_)
            vars_ = set(vars_)
            descriptors = {}
            for e in vars_:
                descriptors[e] = 0
                for i in xrange(len(aggdescriptors_idxs)):
                    if e in aggdescriptors_idxs[i].keys():
                        descriptors[e] += aggdescriptors_idxs[i][e]
                descriptors[e] /= count_vars[e]
    return descriptors
