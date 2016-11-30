
"""
Perturbations
-------------
Module oriented to perform a perturbation of the system in order to carry out
with statistical testing of models.
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
class BasePerturbation:
    """General perturbation. It constains default functions for perturbation
    objects.
    """

    def _initialization(self):
        self.locations_p = None
        self.features_p = None
        self.relations_p = None
        self.discretizations_p = None
        self.k_perturb = 1
        ## Ensure correctness
        self.assert_correctness()

    def assert_correctness(self):
        """Assert the correct Perturbation class."""
        assert('_categorytype' in dir(self))
        assert('_perturbtype' in dir(self))

    def apply2indice(self, i, k):
        """Apply the transformation to the indices.

        Parameters
        ----------
        i: int, list or np.ndarray
            the indices of the elements `i`.
        k: int, list
            the perturbation indices.

        Returns
        -------
        i: int, list or np.ndarray
            the indices of the elements `i`.

        """
        return i

    ################## Transformations of the main elements ###################
    def apply2locs(self, locations):
        """Apply perturbation to locations.

        Parameters
        ----------
        locations: np.ndarray or others
            the spatial information to be perturbed.

        Returns
        -------
        locations: np.ndarray or others
            the spatial information perturbated.

        """
        return locations

    def apply2features(self, features):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        return features

    def apply2relations(self, relations):
        """Apply perturbation to relations.

        Parameters
        ----------
        relations: np.ndarray or others
            the relations between elements to be perturbated.

        Returns
        -------
        relations: np.ndarray or others
            the relations between elements perturbated.

        """
        return relations

    def apply2discretizations(self, discretization):
        """Apply perturbation to discretization.

        Parameters
        ----------
        discretization: np.ndarray or others
            the discretization perturbation.

        Returns
        -------
        discretization: np.ndarray or others
            the discretization perturbation.

        """
        return discretization

    ######################### Precomputed applications ########################
    def apply2features_ind(self, features, i, k):
        """Apply perturbation to features individually for precomputed
        applications.

        Parameters
        ----------
        features: np.ndarray or others
            the element features to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the element features perturbated.

        """
        return self.features_p[i, :, k]

    def apply2locs_ind(self, locations, i, k):
        """Apply perturbation to locations individually for precomputed
        applications.

        Parameters
        ----------
        locations: np.ndarray or others
            the spatial information to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the spatial information perturbated.

        """
        return self.locations_p[i, :, k]

    def apply2relations_ind(self, relations, i, k):
        """For precomputed applications. Apply perturbation to relations.

        Parameters
        ----------
        relations: np.ndarray or others
            the relations between elements to be perturbated.

        Returns
        -------
        relations: np.ndarray or others
            the relations between elements perturbated.

        """
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
class NonePerturbation(BasePerturbation):
    """None perturbation. Default perturbation which not alters the system."""
    _categorytype = "general"
    _perturbtype = "none"

    def __init__(self, k_perturb=1):
        """The none perturbation, null perturbation where anything happens.

        Parameters
        ----------
        k_perturb: int (default=1)
            the number of perturbations applied.

        """
        self._initialization()
        self.k_perturb = k_perturb


###############################################################################
############################ Location perturbation ############################
###############################################################################
class JitterLocations(BasePerturbation):
    """Jitter module to perturbe locations of the system in order of testing
    methods.
    TODO: Fit some model for infering stds.
    """
    _categorytype = "location"
    _perturbtype = "jitter_coordinate"

    def __init__(self, stds=0, k_perturb=1):
        """The jitter locations apply to locations a jittering perturbation.

        Parameters
        ----------
        k_perturb: int (default=1)
            the number of perturbations applied.

        """
        self._initialization()
        self._stds = np.array(stds)
        self.k_perturb = k_perturb

    def apply2locs(self, locations, k=None):
        """Apply perturbation to locations.

        Parameters
        ----------
        locations: np.ndarray
            the spatial information to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray
            the spatial information perturbated.

        """
        ## Preparation of ks
        ks = range(self.k_perturb) if k is None else k
        ks = [k] if type(k) == int else ks
        locations_p = np.zeros((len(locations), locations.shape[1], len(ks)))
        for ik in range(len(ks)):
            jitter_d = np.random.random(locations.shape)
            locations_pj = np.multiply(self._stds, jitter_d) + locations
            locations_p[:, :, ik] = locations_pj
        return locations_p


class PermutationPerturbationLocations(BasePerturbation):
    """Reindice perturbation for the whole locations."""
    _categorytype = "location"
    _perturbtype = "element_permutation"

    def __init__(self, reindices):
        """Perturbations by permuting locations.

        Parameters
        ----------
        reindices: np.ndarray
            the reindices to apply permutation perturbations.

        """
        self._initialization()
        self._format_reindices(reindices)

    def _format_reindices(self, reindices):
        """Format reindices.

        Parameters
        ----------
        reindices: np.ndarray or tuple
            the reindices to apply permutation perturbations.

        """
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
        """Apply perturbation to locations.

        Parameters
        ----------
        locations: np.ndarray
            the spatial information to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray
            the spatial information perturbated.

        """
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
        """Apply the transformation to the indices.

        Parameters
        ----------
        i: int, list or np.ndarray
            the indices of the elements `i`.
        k: int, list
            the perturbation indices.

        Returns
        -------
        i: int, list or np.ndarray
            the indices of the elements `i`.

        """
        return self.reindices[i, k]


###############################################################################
########################### Permutation perturbation ##########################
###############################################################################
class PermutationPerturbation(BasePerturbation):
    """Reindice perturbation for the whole features variables."""
    _categorytype = "feature"
    _perturbtype = "element_permutation"

    def __init__(self, reindices):
        """Element perturbation for all permutation perturbation.

        Parameters
        ----------
        reindices: np.ndarray or tuple
            the reindices to apply permutation perturbations.

        """
        self._initialization()
        self._format_reindices(reindices)

    def _format_reindices(self, reindices):
        """Format reindices for permutation reindices.

        Parameters
        ----------
        reindices: np.ndarray or tuple
            the reindices to apply permutation perturbations.

        """
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
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
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
        """Apply perturbation to features individually for precomputed
        applications.

        Parameters
        ----------
        features: np.ndarray or others
            the element features to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the element features perturbated.

        """
        return features[self.reindices[i, k]]

    def apply2indice(self, i, k):
        """Apply the transformation to the indices.

        Parameters
        ----------
        i: int, list or np.ndarray
            the indices of the elements `i`.
        k: int, list
            the perturbation indices.

        Returns
        -------
        i: int, list or np.ndarray
            the indices of the elements `i`.

        """
        return self.reindices[i, k]


class PermutationPerturbationGeneration(PermutationPerturbation):
    """Reindice perturbation for the whole features variables."""

    def __init__(self, n, m=1, seed=None):
        """Element perturbation for all permutation perturbation.

        Parameters
        ----------
        n: int
            the size of the sample to create the reindices.
        m: int (default=1)
            the number of permutations we want to generate.
        seed: int (default=Npne)
            the seed to initialize and create the same reindices.

        """
        self._initialization()
        if seed is not None:
            np.random.seed(seed)
        self._format_reindices((n, m))


class PartialPermutationPerturbationGeneration(PermutationPerturbation):
    """Reindice perturbation for the whole features variables. It can control
    the proportion of the whole sample is going to be permuted.
    """

    def __init__(self, n, rate_pert=1., m=1, seed=None):
        """Element perturbation for all permutation perturbation.

        Parameters
        ----------
        n: int
            the size of the sample to create the reindices.
        m: int (default=1)
            the number of permutations we want to generate.
        seed: int (default=Npne)
            the seed to initialize and create the same reindices.

        """
        self._initialization()
        if seed is not None:
            np.random.seed(seed)
        if rate_pert == 1.:
            self._format_reindices((n, m))
        else:
            n_sample = int(n*rate_pert)
            indices = np.random.permutation(n)[:n_sample]
            reindices = np.vstack([np.arange(n) for i in xrange(m)]).T
            reindices[indices] = np.vstack([np.random.permutation(n_sample)
                                            for i in xrange(m)]).T
            self.k_perturb = m
            self.reindices = reindices


###############################################################################
############################# Element perturbation ############################
###############################################################################
## TODO:
class MixedFeaturePertubation(BasePerturbation):
    """An individual-column-created perturbation of individual elements."""
    _categorytype = "feature"
    _perturbtype = "element_mixed"

    def __init__(self, perturbations):
        """The MixedFeaturePertubation is the application of different
        perturbations to features.

        perturbations: list
            the list of pst.BasePerturbation objects.

        """
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
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
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
class DiscreteIndPerturbation(BasePerturbation):
    """Discrete perturbation of a discrete feature variable."""
    _categorytype = "feature"
    _perturbtype = "discrete"

    def __init__(self, probs):
        """The discrete individual perturbation to a feature variable.

        Parameters
        ----------
        probs: np.ndarray
            the probabilities to change from a value of a category to another
            value.

        """
        self._initialization()
        if np.all(probs.sum(1) != 1):
            raise TypeError("Not correct probs input.")
        if probs.shape[0] != probs.shape[1]:
            raise IndexError("Probs is noot a square matrix.")
        self.probs = probs.cumsum(1)

    def apply2features(self, feature, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
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


class ContiniousIndPerturbation(BasePerturbation):
    """Continious perturbation for an individual feature variable."""
    _categorytype = "feature"
    _perturbtype = "continious"

    def __init__(self, pstd):
        """The continious individual perturbation to a feature variable.

        Parameters
        ----------
        pstd: float
            the dispersion measure of the jittering.

        """
        self._initialization()
        self.pstd = pstd

    def apply2features(self, feature, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        if k is None:
            k = list(range(self.k_perturb))
        if type(k) == int:
            k = [k]
        feature_p = np.zeros((len(feature), len(k)))
        for i_k in k:
            jitter_d = np.random.random(len(feature))
            feature_p[:, i_k] = np.multiply(self.pstd, jitter_d)
        return feature_p


class PermutationIndPerturbation(BasePerturbation):
    """Reindice perturbation for an individual feature variable."""
    _categorytype = "feature"
    _perturbtype = "permutation_ind"

    def __init__(self, reindices=None):
        """Individual feature perturbation.

        Parameters
        ----------
        reindices: np.ndarray (default=None)
            the reindices to apply permutation perturbations.

        """
        self._initialization()
        if type(reindices) == np.ndarray:
            self.reindices = reindices
            self.k_perturb = reindices.shape[1]
        else:
            raise TypeError("Incorrect reindices.")

    def apply2features(self, feature, k=None):
        """Apply perturbation to features.

        Parameters
        ----------
        features: np.ndarray or others
            the element features collection to be perturbed.
        k: int (default=None)
            the perturbation indices.

        Returns
        -------
        features: np.ndarray or others
            the element features collection perturbated.

        """
        if k is None:
            k = list(range(self.k_perturb))
        if type(k) == int:
            k = [k]
        feature_p = np.zeros((len(feature), len(k)))
        for i_k in k:
            feature_p[:, [i_k]] = feature[self.reindices[:, i_k]]
        return feature_p

    def apply2features_ind(self, feature, i, k):
        """Apply perturbation to features individually for precomputed
        applications.

        Parameters
        ----------
        features: np.ndarray or others
            the element features to be perturbed.
        i: int or list
            the element indices.
        k: int or list
            the perturbation indices.

        Returns
        -------
        locations: np.ndarray or others
            the element features perturbated.

        """
        return feature[self.reindices[i, k]]


###############################################################################
########################### Aggregation perturbation ##########################
###############################################################################
class JitterRelationsPerturbation(BasePerturbation):
    """Jitter module to perturbe relations of the system in order of testing
    methods.
    """
    _categorytype = "relations"
