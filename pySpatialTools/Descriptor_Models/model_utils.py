
"""
Model utils
-----------
Model utils for computing and representing the models correlations.

"""

import numpy as np


def retrieve(kdobject, Coord, i, r):
    """TODEPRECATE: it has to be included in retrieve module.
    """
    results = kdobject.query_ball_point(Coord[i, :], r)
    results.remove(i)
    return results


def filter_with_random_nets(net, random_nets, p_thr):
    """Function to filter the statistically irrelevant correlations between
    variables after comparing with random possible correlations.

    Parameters
    ----------
    net: numpy.ndarrray
        the correlation matrix to filter.
    random_nets: numpy.ndarrray
        the random correlation matrices.
    p_thr: float [0, 1]
        the threshold in percentage.

    Returns
    -------
    net: numpy.ndarrray
        the filtered correlation matrix.

    """
    ## 0. Needed variables
    n = random_nets.shape[2]
    ## 1. Compute bool_net
    bool_net = np.zeros(net.shape)
    for i in range(net.shape[0]):
        for j in range(net.shape[1]):
            bool_net[i, j] = np.sum(net[i, j] < random_nets[i, j, :])/n < p_thr
    net = net*bool_net
    return net


def reorder_net(net, type_vals):
    """Sort the type vals and the net given in order to a better presentation.

    Parameters
    ----------
    net: numpy.ndarrray
        the filtered correlation matrix.
    type_vals: dict
        the list of names of each one of the variables which we have computed
        correlation.

    Returns
    -------
    out: numpy.ndarrray
        the filtered ordered correlation matrix.
    s_type_vals: dict
        sorted type_vals.
    indices: list
        the list the indices which can order the type_vals.

    """
    indices = sorted(range(len(type_vals)), key=lambda k: type_vals[k])
    s_type_vals = sorted(type_vals)
    out = net[indices, :]
    out = out[:, indices]
    return out, s_type_vals, indices
