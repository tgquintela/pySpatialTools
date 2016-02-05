
"""
testing feature retriever and perturbation creation
"""

from pySpatialTools.Feature_engineering.Descriptors import\
    Countdescriptor, AvgDescriptor
from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    CollectionRetrievers
from pySpatialTools.Feature_engineering import SpatialDescriptorModel
from pySpatialTools.Feature_engineering.features_retriever import Features,\
    AggFeatures, FeaturesRetriever, PointFeatures
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    PointFeaturePertubation

import numpy as np


from pySpatialTools.Retrieve.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import OrderEleNeigh, SameEleNeigh
from pySpatialTools.Retrieve.Spatial_Relations import AvgDistanceRegions
import numpy as np
from pySpatialTools.Retrieve import create_retriever_input_output


def test():
    n = 1000
    locs = np.random.random((n, 2))*100
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))
    feat_arr = np.hstack([feat_arr0, feat_arr1])
    features = PointFeatures(feat_arr)

    reindices = np.vstack([np.random.permutation(len(feat_arr))
                          for i in range(5)])
    perturbation = PermutationPerturbation(reindices.T)
    features.add_perturbations(perturbation)

    griddisc = GridSpatialDisc((100, 100), (0, 10), (0, 10))
    locs = np.random.random((n, 2)) * 10
    info_ret = {'order': 4}
    contiguity = griddisc.get_contiguity(out_='sp_relations')
    ret = OrderEleNeigh(contiguity, info_ret)
    regdists = AvgDistanceRegions()
    regdists.compute_distances(locs, griddisc, ret)

    regret = OrderEleNeigh(regdists, {'order': 1})
    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
    regret._output_map = [m_out]

    agg_funct = lambda x, y: x.sum(0).ravel()

    aggfeatures = features.add_aggregations((locs, griddisc), regret,
                                            agg_funct)

    feats_ret = FeaturesRetriever([features, aggfeatures])
