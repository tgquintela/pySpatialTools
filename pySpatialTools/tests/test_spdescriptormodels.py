
"""

"""

## Retrieve
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.SpatialRelations.regiondistances_computers\
    import compute_AvgDistanceRegions
from pySpatialTools.SpatialRelations import RegionDistances

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
import copy

def test():
    n = 1000
    locs = np.random.random((n, 2))*10
    ## Retrievers management
    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, .3, ifdistance=True)
    #countdesc = Countdescriptor()

    # Creation of retriever of regions
    griddisc = GridSpatialDisc((100, 100), (0, 10), (0, 10))
    info_ret = {'order': 4}
    contiguity = griddisc.get_contiguity()
    contiguity = RegionDistances(relations=contiguity)
    ret = OrderEleNeigh(contiguity, info_ret)

    relations, _data, symmetric, store =\
        compute_AvgDistanceRegions(locs, griddisc, ret)
    regdists = RegionDistances(relations=relations, _data=_data,
                               symmetric=symmetric)

    regret = OrderEleNeigh(regdists, {'order': 1})
    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
    regret._output_map = [m_out]
    gret = RetrieverManager([ret0, ret1, regret])

    ## Features management
    feat_arr0 = np.random.randint(0, 20, (n, 1))

    features = ImplicitFeatures(feat_arr0)
    reindices = np.vstack([np.random.permutation(n) for i in range(5)])
    perturbation = PermutationPerturbation(reindices.T)

    features.add_perturbations(perturbation)

    def map_vals_i_t(s, i, k):
        k_p, k_i = s.features[0]._map_perturb(k)
        i_n = s.features[0]._perturbators[k_p].apply2indice(i, k_i)
        return feat_arr0[i_n].ravel()[0]
    map_vals_i = create_mapper_vals_i(map_vals_i_t, feat_arr0)

    countdesc = Countdescriptor()
    feats_ret = FeaturesManager([features], countdesc, maps_vals_i=map_vals_i)
    feats_ret.add_aggregations((locs, griddisc), regret)

#    agg_funct = lambda x, y: x.sum(0).ravel()
#    aggfeatures = feats_ret.add_aggregations((locs, griddisc), regret,
#                                             agg_funct)

    ## Descriptor
    #avgdesc = AvgDescriptor(feats_ret)

    spdesc = SpatialDescriptorModel(gret, feats_ret)
    nets = spdesc.compute()

    spdescs = []
    idxs = slice(0, 250, 1), slice(250, 500, 1), slice(500, 750, 1), slice(750, 1000, 1)
    for i in range(4):
        aux_spdesc = copy.copy(spdesc)
        aux_spdesc.set_loop(idxs[i])
        spdescs.append(aux_spdesc)

    netss = [spdescs[i].compute() for i in range(4)]
