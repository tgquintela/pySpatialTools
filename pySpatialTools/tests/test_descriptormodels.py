
## Retrieve
from pySpatialTools.Retrieve.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve.Spatial_Relations import AvgDistanceRegions
from pySpatialTools.Retrieve import create_retriever_input_output,\
    OrderEleNeigh, SameEleNeigh, KRetriever, CircRetriever,\
    CollectionRetrievers

## Features
from pySpatialTools.Feature_engineering.features_retriever import Features,\
    AggFeatures, FeaturesRetriever, PointFeatures
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    PointFeaturePertubation
from pySpatialTools.IO import create_mapper_vals_i

## Descriptormodel
from pySpatialTools.Feature_engineering.aux_descriptormodels import\
    aggregator_1sh_counter
from pySpatialTools.Feature_engineering.Descriptors import Countdescriptor,\
    AvgDescriptor
from pySpatialTools.Feature_engineering import SpatialDescriptorModel

import numpy as np


def test():
    n = 1000
    locs = np.random.random((n, 2))*100
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))

    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, 3, ifdistance=True)
    gret0 = CollectionRetrievers([ret0])
    gret1 = CollectionRetrievers([ret1])

    def map_vals_i_t(s, i, k):
        k_p, k_i = s.features[0]._map_perturb(k)
        i_n = s.features[0]._perturbators[k_p].apply_reindice(i, k_i)
        return feat_arr0[i_n].ravel()[0]
    map_vals_i = create_mapper_vals_i(feat_arr0, type_sp=map_vals_i_t)

    feats0 = PointFeatures(feat_arr0)
    feats1 = PointFeatures(feat_arr1)
    feats_ret0 = FeaturesRetriever([feats0])
    feats_ret1 = FeaturesRetriever([feats1])
    countdesc = Countdescriptor(feats_ret0, map_vals_i)
    avgdesc = AvgDescriptor(feats_ret1)

    sp_model0 = SpatialDescriptorModel(gret0, avgdesc)
    sp_model1 = SpatialDescriptorModel(gret1, avgdesc)
    sp_model2 = SpatialDescriptorModel(gret0, countdesc)
    sp_model3 = SpatialDescriptorModel(gret1, countdesc)

    corr = sp_model0.compute_nets()
    corr = sp_model1.compute_nets()
    corr = sp_model2.compute_nets()
    corr = sp_model3.compute_nets()
