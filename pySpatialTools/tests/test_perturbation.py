
"""
test Perturbations
------------------
test for perturbations.

"""

import numpy as np

## Retrievers
from pySpatialTools.Retrieve import KRetriever, CircRetriever

## Features
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import Features,\
    ImplicitFeatures, ExplicitFeatures
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    NonePerturbation, JitterLocations, PermutationIndPerturbation,\
    ContiniousIndPerturbation, DiscreteIndPerturbation, MixedFeaturePertubation

from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor


def test():
    n = 1000
    locs = np.random.random((n, 2))*100
    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
    k_perturb4 = k_perturb1+k_perturb2+k_perturb3

    ## Create perturbations
    reind = np.vstack([np.random.permutation(n) for i in range(k_perturb1)])
    perturbation1 = PermutationPerturbation(reind.T)
    perturbation2 = NonePerturbation(k_perturb2)
    perturbation3 = JitterLocations(0.2, k_perturb3)
    perturbation4 = [perturbation1, perturbation2, perturbation3]

    ## Perturbation retrievers
    # Perturbation 1
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation1)
    ret2.add_perturbations(perturbation1)
    assert(ret1.k_perturb == perturbation1.k_perturb)
    assert(ret2.k_perturb == perturbation1.k_perturb)

    # Perturbation 2
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation2)
    ret2.add_perturbations(perturbation2)
    assert(ret1.k_perturb == perturbation2.k_perturb)
    assert(ret2.k_perturb == perturbation2.k_perturb)

    # Perturbation 3
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation3)
    ret2.add_perturbations(perturbation3)
    assert(ret1.k_perturb == perturbation3.k_perturb)
    assert(ret2.k_perturb == perturbation3.k_perturb)

    # Perturbation 4
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation4)
    ret2.add_perturbations(perturbation4)
    assert(ret1.k_perturb == k_perturb4)
    assert(ret2.k_perturb == k_perturb4)

    ## Perturbations features
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))
    feat_arr = np.hstack([feat_arr0, feat_arr1])

    # Perturbation 1
    features = ImplicitFeatures(feat_arr)
    features.add_perturbations(perturbation1)
    avgdesc = AvgDescriptor()
    features = FeaturesManager(features, avgdesc)
    assert(features.k_perturb == perturbation1.k_perturb)

    # Perturbation 2
    features = ImplicitFeatures(feat_arr)
    features.add_perturbations(perturbation2)
    avgdesc = AvgDescriptor()
    features = FeaturesManager(features, avgdesc)
    assert(features.k_perturb == perturbation2.k_perturb)

    # Perturbation 3
    features = ImplicitFeatures(feat_arr)
    features.add_perturbations(perturbation3)
    avgdesc = AvgDescriptor()
    features = FeaturesManager(features, avgdesc)
    assert(features.k_perturb == perturbation3.k_perturb)

    # Perturbation 4
    features = ImplicitFeatures(feat_arr)
    features.add_perturbations(perturbation4)
    avgdesc = AvgDescriptor()
    features = FeaturesManager(features, avgdesc)
    assert(features.k_perturb == k_perturb4)

    ## Individual perturbations
    reind_ind = np.random.permutation(100).reshape((100, 1))
    perm_ind = PermutationIndPerturbation(reind_ind)
    perm_ind.reindices
    feat_perm = np.random.random((100, 1))

    cont_ind = ContiniousIndPerturbation(0.5)
    feat_cont = np.random.random((100, 1))

    disc_ind = DiscreteIndPerturbation(np.random.random((10, 10)))
    feat_disc = np.random.randint(0, 10, 100)

    mix_coll = MixedFeaturePertubation([perm_ind, cont_ind, disc_ind])
    feat_mix = np.hstack([feat_perm, feat_cont, feat_disc.reshape((100, 1))])

    ## TESTING MAIN FUNCTIONS FOR ALL PERTURBATIONS
    perturbation1.apply2indice(0, 0)
    perturbation1.apply2locs(locs)
#    perturbation1.apply2locs_ind(locs, 0, 0)
    perturbation1.selfcompute_locations(locs)
    perturbation1.apply2features(feat_arr)
    perturbation1.apply2features_ind(feat_arr, 0, 0)
    perturbation1.selfcompute_features(feat_arr)

    perturbation2.apply2indice(0, 0)
    perturbation2.apply2locs(locs)
#    perturbation2.apply2locs_ind(locs, 0, 0)
    perturbation2.selfcompute_locations(locs)
    perturbation2.apply2features(feat_arr)
#    perturbation2.apply2features_ind(feat_arr, 0, 0)
    perturbation2.selfcompute_features(feat_arr)

    perturbation3.apply2indice(0, 0)
    perturbation3.apply2locs(locs)
#    perturbation3.apply2locs_ind(locs, 0, 0)
    perturbation3.selfcompute_locations(locs)
    perturbation3.apply2features(feat_arr)
#    perturbation3.apply2features_ind(feat_arr, 0, 0)
    perturbation3.selfcompute_features(feat_arr)

    perm_ind.apply2indice(0, 0)
    perm_ind.apply2locs(locs)
#    perm_ind.apply2locs_ind(locs, 0, 0)
    perm_ind.selfcompute_locations(locs)
    perm_ind.apply2features(feat_perm)
    perm_ind.apply2features_ind(feat_perm, 0, 0)
    perm_ind.selfcompute_features(feat_perm)

    cont_ind.apply2indice(0, 0)
    cont_ind.apply2locs(locs)
#    cont_ind.apply2locs_ind(locs, 0, 0)
    cont_ind.selfcompute_locations(locs)
    cont_ind.apply2features(feat_cont)
#    cont_ind.apply2features_ind(feat_cont, 0, 0)
    cont_ind.selfcompute_features(feat_cont)

    disc_ind.apply2indice(0, 0)
    disc_ind.apply2locs(locs)
#    disc_ind.apply2locs_ind(locs, 0, 0)
    disc_ind.selfcompute_locations(locs)
    disc_ind.apply2features(feat_disc)
#    disc_ind.apply2features_ind(feat_disc, 0, 0)
    disc_ind.selfcompute_features(feat_disc)

    mix_coll.apply2indice(0, 0)
    mix_coll.apply2locs(locs)
#    mix_coll.apply2locs_ind(locs, 0, 0)
    mix_coll.selfcompute_locations(locs)
    mix_coll.apply2features(feat_mix)
#    mix_coll.apply2features_ind(feat_mix, 0, 0)
    mix_coll.selfcompute_features(feat_mix)
