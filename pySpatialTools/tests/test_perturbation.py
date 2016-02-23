
"""
test SpatialRelations
---------------------
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
    NonePerturbation, JitterLocations

from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor

#from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
#    RetrieverManager
#from pySpatialTools.FeatureManagement import SpatialDescriptorModel
#from pySpatialTools.Discretization import GridSpatialDisc
#from pySpatialTools.Retrieve import OrderEleNeigh, SameEleNeigh
#from pySpatialTools.SpatialRelations.regiondistances_computers\
#    import compute_AvgDistanceRegions
#from pySpatialTools.SpatialRelations import RegionDistances
#from pySpatialTools.Retrieve import create_retriever_input_output


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

#    griddisc = GridSpatialDisc((100, 100), (0, 10), (0, 10))
#    locs = np.random.random((n, 2)) * 10
#    info_ret = {'order': 4}
#    contiguity = griddisc.get_contiguity()
#    contiguity = RegionDistances(contiguity)
#    ret = OrderEleNeigh(contiguity, info_ret)
#    relations, _data, symmetric, store =\
#        compute_AvgDistanceRegions(locs, griddisc, ret)
#    regdists = RegionDistances(relations=relations, _data=_data,
#                               symmetric=symmetric)
#    regret = OrderEleNeigh(regdists, {'order': 1})
#    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
#    regret._output_map = [m_out]
#    agg_funct = lambda x, y: x.sum(0).ravel()
#    aggfeatures = features.add_aggregations((locs, griddisc), regret,
#                                            agg_funct)
#
#    feats_ret = FeaturesManager([features, aggfeatures], AvgDescriptor())
