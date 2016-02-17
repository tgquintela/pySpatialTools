
## Retrieve
#from pySpatialTools.Discretization import GridSpatialDisc
#from pySpatialTools.Retrieve.SpatialRelations import AvgDistanceRegions
from pySpatialTools.Retrieve import create_retriever_input_output,\
    OrderEleNeigh, SameEleNeigh, KRetriever, CircRetriever,\
    RetrieverManager

## Features
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import Features,\
    ImplicitFeatures, ExplicitFeatures

from pySpatialTools.utils.perturbations import PermutationPerturbation
from pySpatialTools.utils.util_classes import create_mapper_vals_i

## Descriptormodel
from pySpatialTools.FeatureManagement.aux_descriptormodels import\
    aggregator_1sh_counter
from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor
from pySpatialTools.FeatureManagement import SpatialDescriptorModel

import numpy as np


def test():
    n = 1000
    locs = np.random.random((n, 2))*100
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))

    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, 3, ifdistance=True)
    gret0 = RetrieverManager([ret0])
    gret1 = RetrieverManager([ret1])

    def map_vals_i_t(s, i, k):
        k_p, k_i = s.features[0]._map_perturb(k)
        i_n = s.features[0]._perturbators[k_p].apply2indice(i, k_i)
        return feat_arr0[i_n].ravel()[0]
    map_vals_i = create_mapper_vals_i(map_vals_i_t, feat_arr0)

    feats0 = ImplicitFeatures(feat_arr0)
    feats1 = ImplicitFeatures(feat_arr1)

    avgdesc = AvgDescriptor()
    countdesc = Countdescriptor()

    feats_ret0 = FeaturesManager(feats0, countdesc, maps_vals_i=map_vals_i)
    feats_ret1 = FeaturesManager([feats1], avgdesc, maps_vals_i=map_vals_i)

    sp_model0 = SpatialDescriptorModel(gret0, feats_ret1)
    sp_model1 = SpatialDescriptorModel(gret1, feats_ret1)
    sp_model2 = SpatialDescriptorModel(gret0, feats_ret0)
    sp_model3 = SpatialDescriptorModel(gret1, feats_ret0)

    corr = sp_model0.compute()
    corr = sp_model1.compute()
    corr = sp_model2.compute()
    corr = sp_model3.compute()
