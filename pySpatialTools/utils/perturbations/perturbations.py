
"""
Perturbations
-------------
Module oriented to perform a perturbation of the system in order to carry out
with testing of models.
The main function of this module is grouping functions which are able to
change the system to other statistically probable options in order to explore
the sample space.


TODO
----
-Aggregation perturbation:
--- Discretization perturbed.
--- Fluctuation of features between borders.
- Fluctuation of borders
--- Fluctuation of edge points
--- Fluctuation over sampling points

"""


import numpy as np


###############################################################################
############################ Location perturbation ############################
###############################################################################
class GeneralPerturbation:
    """General perturbation. It constains default functions for perturbation
    objects.
    """

    def _initialization(self):
        self.locations_p = None
        self.features_p = None
        self.relations_p = None
        self.discretizations_p = None
        self.k_perturb = 1

    def apply2indice(self, i, k):
        return i

    ################## Transformations of the main elements ###################
    def apply2locs(self, locations):
        return locations

    def apply2features(self, features):
        return features

    def apply2relations(self, relations):
        return relations

    def apply2discretizations(self, discretization):
        return discretization

    ######################### Precomputed applications ########################
    def apply2features_ind(self, features, i, k):
        """For precomputed applications."""
        return self.features_p[i, :, k]

    def apply2locs_ind(self, locations, i, k):
        """For precomputed applications."""
        return self.locations_p[i, :, k]

    def apply2relations_ind(self, relations, i, k):
        """For precomputed applications."""
        return self.relations_p[i, :, k]

    ##################### Selfcomputation of main elements ####################
    def selfcompute_features(self, features):
        pass

    def selfcompute_locations(self, locations):
        pass

    def selfcompute_relations(self, relations):
        pass

    def selfcompute_discretizations(self, discretizations):
        pass

    ################################# Examples ################################
#    def selfcompute_locations(self, locations):
#        self.locations_p = self.apply2locs(locations)
#
#    def selfcompute_features(self, features):
#        self.features_p = self.apply2features(features)


###############################################################################
############################## None perturbation ##############################
###############################################################################
class NonePerturbation(GeneralPerturbation):
    """None perturbation. Default perturbation which not alters the system."""
    _categorytype = "general"
    _perturbtype = "none"

    def __init__(self, k_perturb=1):
        self._initialization()
        self.k_perturb = k_perturb


###############################################################################
############################ Location perturbation ############################
###############################################################################
class JitterLocations(GeneralPerturbation):
    """Jitter module to perturbe locations of the system in order of testing
    methods.
    TODO: Fit some model for infering stds.
    """
    _categorytype = "location"
    _perturbtype = "jitter_coordinate"

    def __init__(self, stds=0, k_perturb=1):
        self._initialization()
        self._stds = np.array(stds)
        self.k_perturb = k_perturb

    def apply2locs(self, locations, k=None):
        ## Preparation of ks
        ks = range(self.k_perturb) if k is None else k
        ks = [k] if type(k) == int else ks
        locations_p = np.zeros((len(locations), locations.shape[1], len(ks)))
        for ik in range(len(ks)):
            jitter_d = np.random.random(locations.shape)
            locations_pj = np.multiply(self._stds, jitter_d) + locations
            locations_p[:, :, ik] = locations_pj
        return locations_p


class PermutationPerturbationLocations(GeneralPerturbation):
    "Reindice perturbation for the whole locations."
    _categorytype = "location"
    _perturbtype = "element_permutation"

    def __init__(self, reindices):
        self._initialization()
        self._format_reindices(reindices)

    def _format_reindices(self, reindices):
        if type(reindices) == np.ndarray:
            self.k_perturb = reindices.shape[1]
            self.reindices = reindices
        elif type(reindices) == tuple:
            n, k_perturb = reindices
            if type(n) == int and type(k_perturb) == int:
                self.k_perturb = k_perturb
                self.reindices = np.vstack([np.random.permutation(n)
                                            for i in xrange(k_perturb)]).T

    def apply2locs(self, locations, k=None):
        ## Preparation of ks
        ks = range(self.k_perturb) if k is None else k
        ks = [k] if type(k) == int else ks
        ##Be coherent with the input location types
        ndim = 1 if '__len__' not in dir(locations[0]) else len(locations[0])
        if type(locations) == np.ndarray:
            locations_p = np.zeros((len(locations), ndim, len(ks)))
            for ik in range(len(ks)):
                locations_p[:, :, ik] = locations[self.reindices[:, ks[ik]]]
        else:
            locations_p = [[[]]*len(locations)]*len(ks)
            for ik in range(len(ks)):
                for i in range(len(locations)):
                    locations_p[ik][i] = locations[self.reindices[i, ks[ik]]]
        return locations_p

    def apply2indice(self, i, k):
        return self.reindices[i, k]


###############################################################################
########################### Permutation perturbation ##########################
###############################################################################
class PermutationPerturbation(GeneralPerturbation):
    "Reindice perturbation for the whole features variables."
    _categorytype = "feature"
    _perturbtype = "element_permutation"

    def __init__(self, reindices):
        self._initialization()
        self._format_reindices(reindices)

    def _format_reindices(self, reindices):
        if type(reindices) == np.ndarray:
            self.k_perturb = reindices.shape[1]
            self.reindices = reindices
        elif type(reindices) == tuple:
            n, k_perturb = reindices
            if type(n) == int and type(k_perturb) == int:
                self.k_perturb = k_perturb
                self.reindices = np.vstack([np.random.permutation(n)
                                            for i in xrange(k_perturb)]).T

    def apply2features(self, features, k=None):
        ## Assert good features
        assert len(features) == len(self.reindices)
        ## Prepare ks
        ks = range(self.k_perturb) if k is None else k
        ks = [k] if type(k) == int else ks
        ## Computation of new prturbated features
        sh = len(features), features.shape[1], len(ks)
        features_p = np.zeros(sh)
        for ik in range(len(ks)):
            features_p[:, :, ik] = features[self.reindices[:, ks[ik]], :]
        return features_p

    def apply2features_ind(self, features, i, k):
        return features[self.reindices[i, k]]

    def apply2indice(self, i, k):
        return self.reindices[i, k]


###############################################################################
############################# Element perturbation ############################
###############################################################################
## TODO:
class MixedFeaturePertubation(GeneralPerturbation):
    """An individual-column-created perturbation of individual elements."""
    _categorytype = "feature"
    _perturbtype = "element_mixed"

    def __init__(self, perturbations):
        msg = "Perturbations is not a list of individual perturbation methods."
        self._initialization()
        if type(perturbations) != list:
            raise TypeError(msg)
        try:
            self.typefeats = [p._perturbtype for p in perturbations]
            k_perturbs = [p.k_perturb for p in perturbations]
            assert all([k == k_perturbs[0] for k in k_perturbs])
            self.k_perturb = k_perturbs[0]
            self.perturbations = perturbations
        except:
            raise TypeError(msg)

    def apply2features(self, features):
        assert features.shape[1] == len(self.perturbations)
        ## Apply individual perturbation for each features
        features_p, n = [], len(features)
        k_pos = list(range(self.k_perturb))
        for i in range(len(self.perturbations)):
            features_p_k =\
                self.perturbations[i].apply2features(features[:, [i]], k_pos)
            features_p_k = features_p_k.reshape((n, 1, self.k_perturb))
            features_p.append(features_p_k)
        features_p = np.concatenate(features_p, axis=1)
        return features_p


########################### Individual perturbation ###########################
###############################################################################
class DiscreteIndPerturbation(GeneralPerturbation):
    "Discrete perturbation of a discrete feature variable."
    _categorytype = "feature"
    _perturbtype = "discrete"

    def __init__(self, probs):
        self._initialization()
        if np.all(probs.sum(1) != 1):
            raise TypeError("Not correct probs input.")
        if probs.shape[0] != probs.shape[1]:
            raise IndexError("Probs is noot a square matrix.")
        self.probs = probs.cumsum(1)

    def apply2features(self, feature, k=None):
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
        feature_p = np.zeros((len(feature), len(k)))
        for i_k in k:
            for i in xrange(len(feature)):
                r = np.random.random()
                idx = np.where(feature[i] == categories)[0]
                idx2 = np.where(self.probs[idx] > r)[0][0]
                feature_p[i, i_k] = categories[idx2]
        return feature_p


class ContiniousIndPerturbation(GeneralPerturbation):
    "Continious perturbation for an individual feature variable."
    _categorytype = "feature"
    _perturbtype = "continious"

    def __init__(self, pstd):
        self._initialization()
        self.pstd = pstd

    def apply2features(self, feature, k=None):
        if k is None:
            k = list(range(self.k_perturb))
        if type(k) == int:
            k = [k]
        feature_p = np.zeros((len(feature), len(k)))
        for i_k in k:
            jitter_d = np.random.random(len(feature))
            feature_p[:, i_k] = np.multiply(self.pstd, jitter_d)
        return feature_p


class PermutationIndPerturbation(GeneralPerturbation):
    """Reindice perturbation for an individual feature variable."""
    _categorytype = "feature"
    _perturbtype = "permutation_ind"

    def __init__(self, reindices=None):
        self._initialization()
        if type(reindices) == np.ndarray:
            self.reindices = reindices
            self.k_perturb = reindices.shape[1]
        else:
            raise TypeError("Incorrect reindices.")

    def apply2features(self, feature, k=None):
        if k is None:
            k = list(range(self.k_perturb))
        if type(k) == int:
            k = [k]
        feature_p = np.zeros((len(feature), len(k)))
        for i_k in k:
            feature_p[:, [i_k]] = feature[self.reindices[:, i_k]]
        return feature_p

    def apply2features_ind(self, feature, i, k):
        return feature[self.reindices[i, k]]


###############################################################################
########################### Aggregation perturbation ##########################
###############################################################################
class JitterRelationsPerturbation(GeneralPerturbation):
    """Jitter module to perturbe relations of the system in order of testing
    methods.
    """
    _categorytype = "relations"
