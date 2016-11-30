
"""
Artificial features
-------------------
Module which groups artificial data creation of features for testing or other
needings in this package.

"""

import numpy as np


def continuous_array_features(n, n_feats):
    """Array-like continuous features.

    Parameters
    ----------
    n: int
        the number of elements we want to consider.
    n_feats: int
        the number of features we want to consider.

    Returns
    -------
    features: np.ndarray
        the random features we want to compute.

    """
    features = np.random.random((n, n_feats))
    return features


def categorical_array_features(n, n_feats):
    """Array-like categorical features.

    Parameters
    ----------
    n: int
        the number of the elements to create their features.
    n_feats: int
        the number of features.

    Returns
    -------
    features: list
        the random features we want to compute.

    """
    n_feats = [n_feats] if type(n_feats) == int else n_feats
    features = []
    for fea in n_feats:
        features.append(np.random.randint(0, fea, n))
    features = np.stack(features, axis=1)
    return features


def continuous_dict_features(n, n_feats):
    """Listdict-like continuous features.

    Parameters
    ----------
    n: int
        the number of the elements to create their features.
    n_feats: int
        the number of features.

    Returns
    -------
    features: list
        the random features we want to compute.

    """
    features = []
    for i in range(n):
        fea = np.unique(np.random.randint(0, n_feats, n_feats))
        features.append(dict(zip(fea, np.random.random(len(fea)))))
    return features


def categorical_dict_features(n, n_feats):
    """Listdict-like categorical features.

    Parameters
    ----------
    n: int
        the number of the elements to create their features.
    n_feats: int
        the number of features.

    Returns
    -------
    features: list
        the random features we want to compute.

    """
    max_v = np.random.randint(1, n)
    features = []
    for i in range(n):
        fea = np.unique(np.random.randint(0, n_feats, n_feats))
        features.append(dict(zip(fea, np.random.randint(0, max_v, len(fea)))))
    return features


def continuous_agg_array_features(n, n_feats, ks):
    """Array-like continuous aggregated features.

    Parameters
    ----------
    n: int
        the number of the elements to create their features.
    n_feats: int
        the number of features.

    Returns
    -------
    features: np.ndarray
        the random features we want to compute.

    """
    features = []
    for k in range(ks):
        features.append(continuous_array_features(n, n_feats))
    features = np.stack(features, axis=2)
    return features


def categorical_agg_array_features(n, n_feats, ks):
    """Array-like categorical aggregated features.

    Parameters
    ----------
    n: int
        the number of the elements to create their features.
    n_feats: int
        the number of features.
    ks: int
        the number of perturbations.

    Returns
    -------
    features: list
        the random features we want to compute.

    """
    features = []
    for k in range(ks):
        features.append(categorical_array_features(n, n_feats))
    features = np.stack(features, axis=2)
    return features


def continuous_agg_dict_features(n, n_feats, ks):
    """Listdict-like continuous aggregated features.

    Parameters
    ----------
    n: int
        the number of the elements to create their features.
    n_feats: int
        the number of features.
    ks: int
        the number of perturbations.

    Returns
    -------
    features: list
        the random features we want to compute.

    """
    features = []
    for k in range(ks):
        features.append(continuous_dict_features(n, n_feats))
    return features


def categorical_agg_dict_features(n, n_feats, ks):
    """Listdict-like categorical aggregated features.

    Parameters
    ----------
    n: int
        the number of the elements to create their features.
    n_feats: int
        the number of features.
    ks: int
        the number of perturbations.

    Returns
    -------
    features: list
        the random features we want to compute.

    """
    features = []
    for k in range(ks):
        features.append(categorical_dict_features(n, n_feats))
    return features
