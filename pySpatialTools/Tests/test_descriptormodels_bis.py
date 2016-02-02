
"""

"""

## Retrieve
from pySpatialTools.Retrieve.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import OrderEleNeigh, SameEleNeigh
from pySpatialTools.Retrieve.Spatial_Relations import AvgDistanceRegions
from pySpatialTools.Retrieve import create_retriever_input_output
from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    CollectionRetrievers
## Features
from pySpatialTools.Feature_engineering.features_retriever import Features,\
    AggFeatures, FeaturesRetriever, PointFeatures
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    PointFeaturePertubation
## Descriptormodel
from pySpatialTools.Feature_engineering.Descriptors import Countdescriptor,\
    AvgDescriptor
from pySpatialTools.Feature_engineering import SpatialDescriptorModel

import numpy as np


def test():
    n = 10000
    locs = np.random.random((n, 2))*10
    ## Retrievers management
    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, .3, ifdistance=True)
    countdesc = Countdescriptor()

    # Creation of retriever of regions
    griddisc = GridSpatialDisc((100, 100), (0, 10), (0, 10))
    info_ret = {'order': 4}
    contiguity = griddisc.get_contiguity(out_='sp_relations')
    ret = OrderEleNeigh(contiguity, info_ret)
    regdists = AvgDistanceRegions()
    regdists.compute_distances(locs, griddisc, ret)
    regret = OrderEleNeigh(regdists, {'order': 1})
    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
    regret._output_map = [m_out]
    gret = CollectionRetrievers([ret0, ret1, regret])

    ## Features management
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))
    feat_arr = np.hstack([feat_arr0, feat_arr1])
    features = PointFeatures(feat_arr0)
    reindices = np.vstack([np.random.permutation(n) for i in range(5)])
    perturbation = PermutationPerturbation(reindices.T)
    agg_funct = lambda x, y: x.sum(0).ravel()
    aggfeatures = features.add_aggregations((locs, griddisc), regret,
                                            agg_funct)
    feats_ret = FeaturesRetriever([features, aggfeatures])

    ## Descriptor
    #avgdesc = AvgDescriptor(feats_ret)
    countdesc = Countdescriptor(feats_ret)

    ## TODO: add aggfeatures from countdescriptor?
    