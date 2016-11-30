
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
from pySpatialTools.FeatureManagement.features_objects import BaseFeatures,\
    ImplicitFeatures, ExplicitFeatures
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    NonePerturbation, JitterLocations, PermutationIndPerturbation,\
    ContiniousIndPerturbation, DiscreteIndPerturbation, MixedFeaturePertubation
from pySpatialTools.utils.perturbations import BasePerturbation
from pySpatialTools.utils.perturbations import sp_general_filter_perturbations,\
    feat_filter_perturbations, ret_filter_perturbations

from pySpatialTools.FeatureManagement.Descriptors import AvgDescriptor


def test():
    n = 1000
    locs = np.random.random((n, 2))*100
    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
    k_perturb4 = k_perturb1+k_perturb2+k_perturb3

    ## Perturbations features
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))
    feat_arr = np.hstack([feat_arr0, feat_arr1])

    ###########################################################################
    #### GeneralPermutations
    ## Create perturbations
    class DummyPerturbation(BasePerturbation):
        _categorytype = 'feature'
        _perturbtype = 'dummy'

        def __init__(self, ):
            self._initialization()
            self.features_p = np.random.random((10, 10, 10))
            self.locations_p = np.random.random((100, 2, 5))
            self.relations_p = np.random.random((100, 2, 5))
    dummypert = DummyPerturbation()

    # Testing main functions
    dummypert.apply2indice(0, 0)
    dummypert.apply2locs(locs)
    dummypert.apply2locs_ind(locs, 0, 0)
    dummypert.apply2features(feat_arr)
    dummypert.apply2features_ind(feat_arr, 0, 0)
    dummypert.apply2relations(None)
    dummypert.apply2relations_ind(None, 0, 0)
    dummypert.apply2discretizations(None)
    dummypert.selfcompute_features(feat_arr)
    dummypert.selfcompute_locations(locs)
    dummypert.selfcompute_relations(None)
    dummypert.selfcompute_discretizations(None)

    ###########################################################################
    #### Permutations
    ## Create perturbations
    perturbation1 = PermutationPerturbation((n, k_perturb1))
    reind = np.vstack([np.random.permutation(n) for i in range(k_perturb1)])
    perturbation1 = PermutationPerturbation(reind.T)

    # Testing main functions individually
    perturbation1.apply2indice(0, 0)
    perturbation1.apply2locs(locs)
#    perturbation1.apply2locs_ind(locs, 0, 0)
    perturbation1.selfcompute_locations(locs)
    perturbation1.apply2features(feat_arr)
    perturbation1.apply2features_ind(feat_arr, 0, 0)
    perturbation1.selfcompute_features(feat_arr)

    # Perturbations in Retriever
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation1)
    ret2.add_perturbations(perturbation1)
    assert(ret1.k_perturb == perturbation1.k_perturb)
    assert(ret2.k_perturb == perturbation1.k_perturb)

    # Perturbations in Descriptors
#    features = ImplicitFeatures(feat_arr)
#    features.add_perturbations(perturbation1)
#    avgdesc = AvgDescriptor()
#    features = FeaturesManager(features, avgdesc)
#    assert(features.k_perturb == perturbation1.k_perturb)

    ###########################################################################
    #### NonePerturbation
    ## Create perturbations
    perturbation2 = NonePerturbation(k_perturb2)

    # Testing main functions individually
    perturbation2.apply2indice(0, 0)
    perturbation2.apply2locs(locs)
#    perturbation2.apply2locs_ind(locs, 0, 0)
    perturbation2.selfcompute_locations(locs)
    perturbation2.apply2features(feat_arr)
#    perturbation2.apply2features_ind(feat_arr, 0, 0)
    perturbation2.selfcompute_features(feat_arr)

    # Perturbations in Retriever
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation2)
    ret2.add_perturbations(perturbation2)
    assert(ret1.k_perturb == perturbation2.k_perturb)
    assert(ret2.k_perturb == perturbation2.k_perturb)

    # Perturbations in Descriptors
#    features = ImplicitFeatures(feat_arr)
#    features.add_perturbations(perturbation2)
#    avgdesc = AvgDescriptor()
#    features = FeaturesManager(features, avgdesc)
#    assert(features.k_perturb == perturbation2.k_perturb)

    ###########################################################################
    #### JitterPerturbations
    ## Create perturbations
    perturbation3 = JitterLocations(0.2, k_perturb3)

    # Testing main functions individually
    perturbation3.apply2indice(0, 0)
    perturbation3.apply2locs(locs)
#    perturbation3.apply2locs_ind(locs, 0, 0)
    perturbation3.selfcompute_locations(locs)
    perturbation3.apply2features(feat_arr)
#    perturbation3.apply2features_ind(feat_arr, 0, 0)
    perturbation3.selfcompute_features(feat_arr)

    # Perturbations in Retriever
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation3)
    ret2.add_perturbations(perturbation3)
    assert(ret1.k_perturb == perturbation3.k_perturb)
    assert(ret2.k_perturb == perturbation3.k_perturb)

    # Perturbations in Descriptors
#    features = ImplicitFeatures(feat_arr)
#    features.add_perturbations(perturbation3)
#    avgdesc = AvgDescriptor()
#    features = FeaturesManager(features, avgdesc)
#    assert(features.k_perturb == perturbation3.k_perturb)

    ###########################################################################
    #### CollectionPerturbations
    ## Create perturbations
    perturbation4 = [perturbation1, perturbation2, perturbation3]

    # Perturbations in Retriever
    ret1 = KRetriever(locs)
    ret2 = CircRetriever(locs)
    ret1.add_perturbations(perturbation4)
    ret2.add_perturbations(perturbation4)
    assert(ret1.k_perturb == k_perturb4)
    assert(ret2.k_perturb == k_perturb4)

    # Perturbations in Descriptors
#    features = ImplicitFeatures(feat_arr)
#    features.add_perturbations(perturbation4)
#    avgdesc = AvgDescriptor()
#    features = FeaturesManager(features, avgdesc)
#    assert(features.k_perturb == k_perturb4)

    ###########################################################################
    #### IndividualPerturbations
    feat_perm = np.random.random((100, 1))
    feat_disc = np.random.randint(0, 10, 100)
    feat_cont = np.random.random((100, 1))

    ### Reindices individually
    # Individual perturbations
    reind_ind = np.random.permutation(100).reshape((100, 1))

    try:
        boolean = False
        perm_ind = PermutationIndPerturbation(list(reind_ind))
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    perm_ind = PermutationIndPerturbation(reind_ind)
    perm_ind.reindices
    # Testing main functions individually
    perm_ind.apply2indice(0, 0)
    perm_ind.apply2locs(locs)
#    perm_ind.apply2locs_ind(locs, 0, 0)
    perm_ind.selfcompute_locations(locs)
    perm_ind.apply2features(feat_perm)
    perm_ind.apply2features(feat_perm, 0)
    perm_ind.apply2features_ind(feat_perm, 0, 0)
    perm_ind.selfcompute_features(feat_perm)

    ### Continious individually
    cont_ind = ContiniousIndPerturbation(0.5)
    # Testing main functions individually
    cont_ind.apply2indice(0, 0)
    cont_ind.apply2locs(locs)
#    cont_ind.apply2locs_ind(locs, 0, 0)
    cont_ind.selfcompute_locations(locs)
    cont_ind.apply2features(feat_cont)
    cont_ind.apply2features(feat_cont, 0)
#    cont_ind.apply2features_ind(feat_cont, 0, 0)
    cont_ind.selfcompute_features(feat_cont)

    ### Discrete individually
    try:
        boolean = False
        disc_ind = DiscreteIndPerturbation(np.random.random((10, 10)))
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        probs = np.random.random((10, 10))
        probs = (probs.T/probs.sum(1)).T
        disc_ind = DiscreteIndPerturbation(probs[:8, :])
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    probs = np.random.random((10, 10))
    probs = (probs.T/probs.sum(1)).T
    disc_ind = DiscreteIndPerturbation(probs)
    # Testing main functions individually
    disc_ind.apply2indice(0, 0)
    disc_ind.apply2locs(locs)
#    disc_ind.apply2locs_ind(locs, 0, 0)
    disc_ind.selfcompute_locations(locs)
    disc_ind.apply2features(feat_disc)
    disc_ind.apply2features(feat_disc, 0)
#    disc_ind.apply2features_ind(feat_disc, 0, 0)
    disc_ind.selfcompute_features(feat_disc)
    try:
        boolean = False
        disc_ind.apply2features(np.random.randint(0, 40, 1000))
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ### Mix individually
    mix_coll = MixedFeaturePertubation([perm_ind, cont_ind, disc_ind])
    # Testing main functions individually
    feat_mix = np.hstack([feat_perm, feat_cont, feat_disc.reshape((100, 1))])
    mix_coll.apply2indice(0, 0)
    mix_coll.apply2locs(locs)
#    mix_coll.apply2locs_ind(locs, 0, 0)
    mix_coll.selfcompute_locations(locs)
    mix_coll.apply2features(feat_mix)
#    mix_coll.apply2features_ind(feat_mix, 0, 0)
    mix_coll.selfcompute_features(feat_mix)

    try:
        boolean = False
        MixedFeaturePertubation(None)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        MixedFeaturePertubation([None])
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        mix_coll.apply2features(None)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        mix_coll.apply2features([None])
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ###########################################################################
    ##################### Auxiliar perturbation functions #####################
    ###########################################################################
    sp_general_filter_perturbations(perturbation1)
    feat_filter_perturbations(perturbation1)
    ret_filter_perturbations(perturbation1)
    sp_general_filter_perturbations(perturbation2)
    feat_filter_perturbations(perturbation2)
    ret_filter_perturbations(perturbation2)
    sp_general_filter_perturbations(perturbation3)
    feat_filter_perturbations(perturbation3)
    ret_filter_perturbations(perturbation3)
    sp_general_filter_perturbations([perturbation1])
    feat_filter_perturbations([perturbation1])
    ret_filter_perturbations([perturbation1])
    sp_general_filter_perturbations([perturbation2])
    feat_filter_perturbations([perturbation2])
    ret_filter_perturbations([perturbation2])
    sp_general_filter_perturbations([perturbation3])
    feat_filter_perturbations([perturbation3])
    ret_filter_perturbations([perturbation3])

    perts = [PermutationPerturbation((n, 5)), NonePerturbation(5),
             JitterLocations(0.2, 5)]

    sp_general_filter_perturbations(perts)
    feat_filter_perturbations(perts)
    ret_filter_perturbations(perts)
