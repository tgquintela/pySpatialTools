
"""
testing feature retriever and perturbation creation
"""

from pySpatialTools.FeatureManagement.Descriptors import\
    Countdescriptor, AvgDescriptor
from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    RetrieverManager
from pySpatialTools.FeatureManagement import SpatialDescriptorModel

## Features
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import Features,\
    ImplicitFeatures, ExplicitFeatures

from pySpatialTools.utils.perturbations import PermutationPerturbation

from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor
import numpy as np


from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import OrderEleNeigh, SameEleNeigh
from pySpatialTools.SpatialRelations.regiondistances_computers\
    import compute_AvgDistanceRegions
from pySpatialTools.SpatialRelations import RegionDistances
from pySpatialTools.Retrieve import create_retriever_input_output


def test():
    n = 1000
    locs = np.random.random((n, 2))*100
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))
    feat_arr = np.hstack([feat_arr0, feat_arr1])
    features = ImplicitFeatures(feat_arr)

    reindices = np.vstack([np.random.permutation(len(feat_arr))
                          for i in range(5)])
    perturbation = PermutationPerturbation(reindices.T)
    features.add_perturbations(perturbation)

    griddisc = GridSpatialDisc((100, 100), (0, 10), (0, 10))
    locs = np.random.random((n, 2)) * 10
    info_ret = {'order': 4}
    contiguity = griddisc.get_contiguity()
    contiguity = RegionDistances(contiguity)

    ret = OrderEleNeigh(contiguity, info_ret)

    relations, _data, symmetric, store =\
        compute_AvgDistanceRegions(locs, griddisc, ret)
    regdists = RegionDistances(relations=relations, _data=_data,
                               symmetric=symmetric)
    regret = OrderEleNeigh(regdists, {'order': 1})
    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
    regret._output_map = [m_out]

    agg_funct = lambda x, y: x.sum(0).ravel()
    avgdesc = AvgDescriptor()
    features = FeaturesManager(features, avgdesc)

#    aggfeatures = features.add_aggregations((locs, griddisc), regret,
#                                            agg_funct)
#
#    feats_ret = FeaturesManager([features, aggfeatures], AvgDescriptor())
