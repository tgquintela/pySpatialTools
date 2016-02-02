
"""
Perturbations
-------------
Module oriented to perform a perturbation of the system in order to carry out
with testing of models.


TODO
----
-Aggregation perturbation:
--- Discretization perturbed.
--- Fluctuation of features between borders.

"""


import numpy as np


###############################################################################
############################ Location perturbation ############################
###############################################################################
class Jitter:
    """Jitter module to perturbe locations of the system in order of testing
    methods.

    TODO: Fit some model for infering stds.

    """
    _stds = 0

    def __init(self, stds):
        stds = np.array(stds)

    def apply(self, coordinates):
        jitter_d = np.random.random(coordinates.shape)
        new_coordinates = np.multiply(self._stds, jitter_d)
        return new_coordinates


###############################################################################
############################# Element perturbation ############################
###############################################################################
class PointFeaturePertubation:
    "An individual column perturbation of individual elements."
    _perturbtype = "point_mixed"
    k_perturb = 0

    def __init__(self, perturbations):
        if type(perturbations) != list:
            msg = "Perturbations is not a list of perturbation methods."
            raise TypeError(msg)
        try:
            self.typefeats = [p._perturbtype for p in perturbations]
            self.perturbations = perturbations
        except:
            msg = "Perturbations is not a list of perturbation methods."
            raise TypeError(msg)

    def apply(self, features):
        assert features.shape[1] == len(self.perturbations)
        ## Apply individual perturbation for each features
        features_p, n = [], len(features)
        k_pos = list(range(self.k_perturb))
        for i in range(len(self.perturbations)):
            features_p_k = self.perturbations[i].apply(features[:, i], k_pos)
            features_p_k = features_p_k.reshape((n, 1, self.k_perturb))
            features_p.append(features_p_k)
        features_p = np.concatenate(features_p, axis=1)
        return features_p

    def selfcompute(self, features):
        self.features_p = self.apply(features)

    def apply_ind(self, features, i, k):
        return self.features_p[i, :, k]


class PermutationPerturbation:
    "Reindice perturbation for the whole features variables."
    _perturbtype = "point_permutation"
    k_perturb = 0

    def __init__(self, reindices):
        self._format_reindices(reindices)

    def apply(self, features):
        assert len(features) == len(self.reindices)
        sh = len(features), features.shape[1], self.reindices.shape[1]
        features_p = np.zeros(sh)
        for i in range(sh[2]):
            features_p[:, :, i] = features[self.reindices[:, i], :]
        return features_p

    def _format_reindices(self, reindices):
        self.k_perturb = reindices.shape[1]
        self.reindices = reindices

    def selfcompute(self, features):
        pass

    def apply_ind(self, features, i, k):
        return features[self.reindices[i, k], :]


class DiscreteIndPerturbation:
    "Discrete perturbation of a discrete feature variable."
    _perturbtype = "discrete"
    k_perturb = 0

    def __init__(self, probs):
        if np.all(probs.sum(1)) != 1:
            raise TypeError("Not correct probs input.")
        if probs.shape[0] != probs.shape[1]:
            raise IndexError("Probs is noot a square matrix.")
        self.probs = probs.cumsum(1)

    def apply(self, feature, k=None):
        ## Prepare loop
        categories = np.unique(feature)
        if len(categories) != len(self.probs):
            msg = "Not matching dimension between probs and features."
            raise IndexError(msg)
        if k is None:
            k = list(range(self.k_perturb))
        if type(k) == int:
            k = [k]
        ## Compute each change
        feature_p = np.zeros((len(feature, len(k))))
        for i_k in k:
            for i in xrange(len(feature)):
                r = np.random.random()
                idx = np.where(feature[i] == categories)[0]
                idx2 = np.where(self.probs[idx] > r)[0][0]
                feature_p[i, i_k] = categories[idx2]
        return feature_p


class ContiniousIndPerturbation:
    "Continious perturbation for an individual feature variable."
    _perturbtype = "continious"
    k_perturb = 0

    def __init__(self, pstd):
        self.pstd = pstd

    def apply(self, feature, k=None):
        if k is None:
            k = list(range(k))
        if type(k) == int:
            k = [k]
        feature_p = np.zeros((len(feature, len(k))))
        for i_k in k:
            jitter_d = np.random.random(len(feature))
            feature_p[:, i_k] = np.multiply(self.pstd, jitter_d)
        return feature_p


class PermutationIndPerturbation:
    "Reindice perturbation for an individual feature variable."
    _perturbtype = "permutation_ind"
    k_perturb = 0

    def __init__(self, reindices=None):
        if type(reindices) == np.array:
            self.reindices = reindices
            self.k_perturb = reindices.shape[1]

    def apply(self, feature, k=None):
        if k is None:
            k = list(range(self.k_perturb))
        if type(k) == int:
            k = [k]
        feature_p = np.zeros((len(feature), len(k)))
        for i_k in k:
            feature_p[:, i_k] = feature[self.reindices[:, i_k]]
        return feature_p

    def apply_ind(self, feature, i, k):
        return feature[self.reindices[i, k]]

###############################################################################
########################### Aggregation perturbation ##########################
###############################################################################
