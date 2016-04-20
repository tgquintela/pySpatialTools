
"""
Artificial features
-------------------
Module which groups artificial data creation of features for testing or other
needings in this package.

"""

import numpy as np


def continuous_array_features(n, n_feats):
    """Array-like continuous features."""
    features = np.random.random((n, n_feats))
    return features


def categorical_array_features(n, n_feats):
    """Array-like categorical features."""
    n_feats = [n_feats] if type(n_feats) == int else n_feats
    features = []
    for fea in n_feats:
        features.append(np.random.randint(0, fea, n))
    features = np.stack(features, axis=1)
    return features


def continuous_dict_features(n, n_feats):
    """Listdict-like continuous features."""
    n_feats = [n_feats] if type(n_feats) == int else n_feats
    features = []
    for fea in n_feats:
        features.append(np.random.randint(0, fea, n))
    features = np.stack(features, axis=1)
    return features


def categorical_dict_features(n, n_feats):
    """Listdict-like categorical features."""
    n_feats = [n_feats] if type(n_feats) == int else n_feats
    features = []
    for fea in n_feats:
        features.append(np.random.randint(0, fea, n))
    features = np.stack(features, axis=1)
    return features


def continuous_agg_array_features(n, n_feats, ks):
    """Array-like continuous aggregated features."""
    features = []
    for k in range(ks):
        features.append(continuous_array_features(n, n_feats))
    features = np.stack(features, axis=2)
    return features


def categorical_agg_array_features(n, n_feats, ks):
    """Array-like categorical aggregated features."""
    features = []
    for k in range(ks):
        features.append(categorical_array_features(n, n_feats))
    features = np.stack(features, axis=2)
    return features


def continuous_agg_dict_features(n, n_feats, ks):
    """Listdict-like continuous aggregated features."""
    features = []
    for k in range(ks):
        features.append(continuous_dict_features(n, n_feats))
    return features


def categorical_agg_dict_features(n, n_feats, ks):
    """Listdict-like categorical aggregated features."""
    features = []
    for k in range(ks):
        features.append(categorical_dict_features(n, n_feats))
    return features
