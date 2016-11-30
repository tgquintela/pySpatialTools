
"""
artificial measure
------------------
Creation of artificial measure
"""

import numpy as np


############################### Create measure ################################
###############################################################################
def create_artificial_measure_array(n_k, n_vals_i, n_feats):
    """Create artificial random measure in the array form.

    Parameters
    ----------
    n_k: int
        the number of perturbations
    n_vals_i: int
        the number of indices of the output measure.
    n_feats: int
        the number of features.

    Returns
    -------
    measure: np.ndarray
        the transformed measure computed by the whole spatial descriptor model.

    """
    measure = np.random.random((n_vals_i, n_feats, n_k))
    return measure


def create_artificial_measure_append(n_k, n_vals_i, n_feats):
    """Create artificial random measure in the list form.

    Parameters
    ----------
    n_k: int
        the number of perturbations
    n_vals_i: int
        the number of indices of the output measure.
    n_feats: int
        the number of features.

    Returns
    -------
    measure: list
        the transformed measure computed by the whole spatial descriptor model.

    """
    rounds = np.random.randint(1, 40)
    measure = create_empty_append(n_k, n_vals_i, n_feats)
    for i in range(rounds):
        n_iss = np.random.randint(1, 10)
        vals_i = create_vals_i(n_iss, n_vals_i, n_k)
        x_i = create_features_i_dict(n_feats, n_iss, n_k)
        for k in range(len(vals_i)):
            for i in range(len(vals_i[k])):
                measure[k][vals_i[k][i]].append(x_i[k][i])
    return measure


def create_artificial_measure_replacelist(n_k, n_vals_i, n_feats,
                                          unique_=False):
    """Create artificial random measure in the replacelist form.

    Parameters
    ----------
    n_k: int
        the number of perturbations
    n_vals_i: int
        the number of indices of the output measure.
    n_feats: int
        the number of features.
    unique_: boolean (default=False)
        if there are no collapse.

    Returns
    -------
    measure: list
        the transformed measure computed by the whole spatial descriptor model.

    """
    last = 0
    rounds = np.random.randint(1, 40)
    measure = create_empty_replacelist(n_k, n_vals_i, n_feats)
    for i in range(rounds):
        n_iss = np.random.randint(1, 10)
        if unique_:
            vals_i = np.array([last+np.arange(n_iss)]*n_k)
            last += n_iss
        else:
            vals_i = create_vals_i(n_iss, n_vals_i, n_k)
        x_i = create_features_i_dict(n_feats, n_iss, n_k)
        for k in range(len(vals_i)):
            measure[k][0].append(x_i[k])
            measure[k][1].append(vals_i[k])
    return measure


############################### Empty measure #################################
###############################################################################
def create_empty_array(n_k, n_vals_i, n_feats):
    """Create null measure in the array form.

    Parameters
    ----------
    n_k: int
        the number of perturbations
    n_vals_i: int
        the number of indices of the output measure.
    n_feats: int
        the number of features.

    Returns
    -------
    measure: np.ndarray
        the null measure to be fill by the computation of the spatial
        descriptor model.

    """
    return np.zeros((n_vals_i, n_feats, n_k))


def create_empty_append(n_k, n_iss, n_feats):
    """Create null measure in the list form.

    Parameters
    ----------
    n_k: int
        the number of perturbations
    n_vals_i: int
        the number of indices of the output measure.
    n_feats: int
        the number of features.

    Returns
    -------
    measure: list
        the null measure to be fill by the computation of the spatial
        descriptor model.

    """
    return [[[]]*n_iss]*n_k


def create_empty_replacelist(n_k, n_iss, n_feats):
    """Create null measure in the replacelist form.

    Parameters
    ----------
    n_k: int
        the number of perturbations
    n_vals_i: int
        the number of indices of the output measure.
    n_feats: int
        the number of features.

    Returns
    -------
    measure: list
        the null measure to be fill by the computation of the spatial
        descriptor model.

    """
    return [[[], []]]*n_k


############################### Vals_i creation ###############################
###############################################################################
def create_vals_i(n_iss, nvals, n_k):
    """

    Parameters
    ----------
    n_k: int
        the number of perturbations
    n_vals_i: int
        the number of indices of the output measure.
    n_feats: int
        the number of features.

    Returns
    -------
    vals_i: np.ndarray
        the associated stored indices for the element indices.

    """
    return np.random.randint(1, nvals, n_iss*n_k).reshape((n_k, n_iss))


############################### Empty features ################################
###############################################################################
def create_empty_features_array(n_feats, n_iss, n_k):
    """Create null features for different iss in an array-form.

    Parameters
    ----------
    n_feats: int
        the number of features.
    n_iss: int
        the number of the elements to create their features.
    n_k: int
        the number of perturbations.

    Returns
    -------
    features: np.ndarray
        the null features we want to compute.

    """
    return np.zeros((n_k, n_iss, n_feats))


def create_empty_features_dict(n_feats, n_iss, n_k):
    """Create null features for different iss in an listdict-form.

    Parameters
    ----------
    n_feats: int
        the number of features.
    n_iss: int
        the number of the elements to create their features.
    n_k: int
        the number of perturbations.

    Returns
    -------
    features: list
        the null features we want to compute.

    """
    return [[{}]*n_iss]*n_k


################################ X_i features #################################
###############################################################################
def create_features_i_array(n_feats, n_iss, n_k):
    """Create null features for different iss in an array-form.

    Parameters
    ----------
    n_feats: int
        the number of features.
    n_iss: int
        the number of the elements to create their features.
    n_k: int
        the number of perturbations.

    Returns
    -------
    features: np.ndarray
        the null features we want to compute.

    """
    x_i = np.random.random((n_k, n_iss, n_feats))
    return x_i


def create_features_i_dict(n_feats, n_iss, n_k):
    """Create null features for different iss in an listdict-form.

    Parameters
    ----------
    n_feats: int
        the number of features.
    n_iss: int
        the number of the elements to create their features.
    n_k: int
        the number of perturbations.

    Returns
    -------
    features: list
        the null features we want to compute.

    """
    x_i = []
    for k in range(n_k):
        x_i_k = []
        for i in range(n_iss):
            keys = np.unique(np.random.randint(1, n_feats, n_feats))
            keys = [str(e) for e in keys]
            values = np.random.random(len(keys))
            x_i_k.append(dict(zip(keys, values)))
        x_i.append(x_i_k)
    return x_i
