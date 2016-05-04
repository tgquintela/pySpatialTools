
"""
test_spdescriptormodels
-----------------------
testing spatial descriptor models utilities.

"""


import numpy as np
import os
import copy

## Retrieve
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import create_retriever_input_output,\
    SameEleNeigh, KRetriever, CircRetriever, RetrieverManager,\
    WindowsRetriever
# Artificial data
from pySpatialTools.utils.artificial_data import generate_randint_relations

## Features
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import Features,\
    ImplicitFeatures, ExplicitFeatures

from pySpatialTools.utils.perturbations import PermutationPerturbation
from pySpatialTools.utils.util_classes import create_mapper_vals_i

## Descriptormodel
from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor, NBinsHistogramDesc, SparseCounter
from pySpatialTools.FeatureManagement import SpatialDescriptorModel

from ..utils.artificial_data import create_random_image
from ..utils.util_external.Logger import Logger
from ..io.io_images import create_locs_features_from_image


def test():
    n, nx, ny = 100, 100, 100
    locs = np.random.random((n, 2))*10
    ## Retrievers management
    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, .3, ifdistance=True)
    #countdesc = Countdescriptor()

    ###########################################################################
    ###########################################################################
    ######## Testing aggregation
    # Creation of retriever of regions
    griddisc = GridSpatialDisc((nx, ny), (0, 10), (0, 10))
    regdists = generate_randint_relations(0.01, (nx, ny), p0=0., maxvalue=1)
    
    
#    regret = SameEleNeigh(regdists, bool_input_idx=False)
#    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
#    regret._output_map = [m_out]
#    gret = RetrieverManager([ret0, ret1, regret])
#    regret = SameEleNeigh(regdists, bool_input_idx=False)

    ## Features management
    feat_arr0 = np.random.randint(0, 20, (n, 1))

    features = ImplicitFeatures(feat_arr0)
    reindices = np.vstack([np.random.permutation(n) for i in range(5)])
    perturbation = PermutationPerturbation(reindices.T)

    features.add_perturbations(perturbation)


#
#    ## Create MAP VALS (indices)
#    corr_arr = -1*np.ones(n)
#    for i in range(len(np.unique(feat_arr0))):
#        corr_arr[(feat_arr0 == np.unique(feat_arr0)[i]).ravel()] = i
#    assert(np.sum(corr_arr == (-1)) == 0)
#
#    def map_vals_i_t(s, i, k):
#        k_p, k_i = s.features[0]._map_perturb(k)
#        i_n = s.features[0]._perturbators[k_p].apply2indice(i, k_i)
#        return corr_arr[i_n]
#    map_vals_i = create_mapper_vals_i(map_vals_i_t, feat_arr0)
#
#    countdesc = Countdescriptor()
#    feats_ret = FeaturesManager([features], countdesc, maps_vals_i=map_vals_i)
#    feats_ret.add_aggregations((locs, griddisc), regret)
#
#    ## Descriptor
#    #avgdesc = AvgDescriptor(feats_ret)
#    spdesc = SpatialDescriptorModel(gret, feats_ret)
#    nets = spdesc.compute()
#    spdescs = []
#    idxs = [slice(0, 25, 1), slice(25, 50, 1), slice(50, 75, 1)]
#    idxs += [slice(75, 100, 1)]
#    for i in range(4):
#        aux_spdesc = copy.copy(spdesc)
#        aux_spdesc.set_loop(idxs[i])
#        spdescs.append(aux_spdesc)
#    netss = [spdescs[i].compute() for i in range(4)]
#
#    try:
#        logfile = Logger('logfile.log')
#        spdesc.compute_process(logfile, lim_rows=0, n_procs=0)
#        os.remove('logfile.log')
#    except:
#        raise Exception("Not usable compute_process.")
#
#    ## Grid descriptors
#    im_example = create_random_image((20, 20))[:, :, 0]
#    shape = im_example.shape
#    feats = im_example.ravel()
#    #locs, feats = create_locs_features_from_image(im_example)
#    pars_ret, nbins = {'l': 8, 'center': 0, 'excluded': False}, 5
#    #windret = WindowsRetriever((10, 10), pars_ret)
#    windret = WindowsRetriever(shape, pars_ret)
#    binsdesc = NBinsHistogramDesc(nbins)
#    features = binsdesc.set_global_info(feats, transform=True)
#
#    gret = RetrieverManager(windret)
#    feats_ret = FeaturesManager(features, binsdesc)
#    spdesc = SpatialDescriptorModel(gret, feats_ret)
##    net = spdesc.compute()
#
#    pars_ret2 = {'l': np.array([8, 3]), 'center': np.array([0, 0]),
#                 'excluded': True}
#    locs, _ = create_locs_features_from_image(im_example)
#    windret2 = WindowsRetriever(locs, pars_ret2)
#    gret2 = RetrieverManager(windret2)
#    spdesc2 = SpatialDescriptorModel(gret2, feats_ret)
##    net2 = spdesc2.compute()
#    try:
#        windret3 = WindowsRetriever((8, 9.), pars_ret2)
#        raise
#    except:
#        pass
#
#
#    ## Categorical array with different windows and dict
#    cat_ts = np.random.randint(0, 20, 20)
#    pars_ret, nbins = {'l': 8, 'center': 0, 'excluded': False}, 5
#    windret = WindowsRetriever(cat_ts.shape, pars_ret)
#    gret = RetrieverManager(windret)
#
#    countdesc = SparseCounter()
#    feats_ret = FeaturesManager(cat_ts, countdesc, out='dict',
#                                maps_vals_i=cat_ts)
#    spdesc = SpatialDescriptorModel(gret, feats_ret)
#    net = spdesc.compute()
#
#    feats_ret = FeaturesManager(cat_ts, countdesc, out='dict')
#    spdesc = SpatialDescriptorModel(gret, feats_ret)
#    net = spdesc.compute()
#
#
#    pars_ret, nbins = {'l': 6, 'center': -1, 'excluded': False}, 5
#    windret = WindowsRetriever(cat_ts.shape, pars_ret)
#    gret = RetrieverManager(windret)
#
#    feats_ret = FeaturesManager(cat_ts, countdesc, out='dict',
#                                maps_vals_i=cat_ts)
#    spdesc = SpatialDescriptorModel(gret, feats_ret)
#    net = spdesc.compute()
