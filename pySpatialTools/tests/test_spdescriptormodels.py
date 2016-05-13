
"""
test_spdescriptormodels
-----------------------
testing spatial descriptor models utilities.

"""

import numpy as np
#import os
import copy
from itertools import product

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
from pySpatialTools.FeatureManagement.features_objects import\
    ImplicitFeatures, ExplicitFeatures

from pySpatialTools.utils.perturbations import PermutationPerturbation
from pySpatialTools.utils.util_classes import create_mapper_vals_i

## Descriptormodel
from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor, NBinsHistogramDesc, SparseCounter
from pySpatialTools.FeatureManagement import SpatialDescriptorModel

#from ..utils.artificial_data import create_random_image
from ..utils.util_external.Logger import Logger
from ..io.io_images import create_locs_features_from_image


def test():
    n, nx, ny = 100, 100, 100
    m, rei = 3, 5
    locs = np.random.random((n, 2))*10
    ## Retrievers management
    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, .3, ifdistance=True)
    #countdesc = Countdescriptor()

    ## Possible feats
    aggfeats = np.random.random((n/2, m, rei))
    featsarr0 = np.random.random((n, m))
    featsarr1 = np.random.random((n, m))
    featsarr2 = np.vstack([np.random.randint(0, 10, n) for i in range(m)]).T
    reindices0 = np.arange(n)
    reindices = np.vstack([reindices0]+[np.random.permutation(n)
                                        for i in range(rei-1)]).T
    perturbation = PermutationPerturbation(reindices)

    feats0 = ExplicitFeatures(aggfeats)
    feats1 = ImplicitFeatures(featsarr0)
    feats2 = FeaturesManager(ExplicitFeatures(aggfeats))

    ## Other functions
    def map_indices(s, i):
        if s._pos_inputs is not None:
            return s._pos_inputs.start + s._pos_inputs.step*i
        else:
            return i

    ## Random exploration functions
    def random_pos_space_exploration(pos_possibles):
        selected, indices = [], []
        for i in range(len(pos_possibles)):
            sel, ind = random_pos_exploration(pos_possibles[i])
            selected.append(sel)
            indices.append(ind)
        return selected, indices

    def random_pos_exploration(possibles):
        ## Selection
        i_pos = np.random.randint(0, len(possibles))
        return possibles[i_pos], i_pos

    ## Impossibles
    def impossible_instantiation(selected, p):
        ret, sel, agg, pert = p
        p_ind, m_ind, n_desc, feat = selected
        checker = False

        ## Not implemented perturbation over explicit features
        if pert is not None:
            if type(feat) == np.ndarray:
                if len(feat.shape) == 3:
                    checker = True
            elif isinstance(feat, ExplicitFeatures):
                checker = True
            elif isinstance(feat, FeaturesManager):
                check_aux = []
                for i in range(len(feat.features)):
                    check_aux.append(isinstance(feat.features[i],
                                                ExplicitFeatures))
                checker = any(check_aux)
        return checker

    def compulsary_instantiation_errors(selected, p):
        ret, sel, agg, pert = p
        p_ind, m_ind, n_desc, feat = selected
        checker = False

        ## Compulsary failing instantiation
        if not checker:
            return
        try:
            boolean = False
            SpatialDescriptorModel(retrievers=ret, featurers=feat,
                                   mapselector_spdescriptor=sel,
                                   pos_inputs=p_ind, map_indices=m_ind,
                                   perturbations=pert, aggregations=agg,
                                   name_desc=n_desc)
            boolean = True
        except:
            if boolean:
                raise Exception("It has to halt here.")

    ###########################################################################
    ###########################################################################
    ######## Testing instantiation spdesc
    # Locs and retrievers
    locs_input = np.random.random((100, 2))
    locs1 = np.random.random((50, 2))
    locs2 = np.random.random((70, 2))
    ret0 = KRetriever(locs1, autolocs=locs_input, info_ret=3)
    ret1 = [ret0, CircRetriever(locs2, info_ret=0.1, autolocs=locs_input)]
    ret2 = RetrieverManager(ret0)
    pos_rets = [ret0, ret1, ret2]
    # Feats and manager

    arrayselector = None
    functselector = None
    listselector = None
    selobj = None
    pos_selectors = [None]
    pos_agg = [None]

    ## Perturbations
    pos_pert = [None, perturbation]

    ## Random exploration
    pos_loop_ind = [None, 20, (0, 100, 1), slice(0, 100, 1)]
    pos_loop_mapin = [None, map_indices]
    pos_name_desc = [None, '', 'random_desc']
    # Possible feats
    pos_feats = [feats0, feats1, aggfeats, featsarr0, feats2]
    # Random exploration possibilities
    pos_random = [pos_loop_ind, pos_loop_mapin, pos_name_desc, pos_feats]

    possibilities = [pos_rets, pos_selectors, pos_agg, pos_pert]

    for p in product(*possibilities):
        ret, sel, agg, pert = p
        ## Random exploration of parameters
        selected, indices = random_pos_space_exploration(pos_random)
        print indices
#        print p, selected
        p_ind, m_ind, n_desc, feat = selected
        ## Impossible cases
        checker = impossible_instantiation(selected, p)
        if checker:
            continue
        ## Testing instantiation
        spdesc = SpatialDescriptorModel(retrievers=ret, featurers=feat,
                                        mapselector_spdescriptor=sel,
                                        pos_inputs=p_ind, map_indices=m_ind,
                                        perturbations=pert, aggregations=agg,
                                        name_desc=n_desc)
        ## Function testing
        spdesc.add_perturbations(pert)
        spdesc.set_loop(p_ind, m_ind)


#    ###########################################################################
#    ###########################################################################
#    ######## Testing aggregation
#    # Creation of retriever of regions
#    griddisc = GridSpatialDisc((nx, ny), (0, 10), (0, 10))
#    regdists = generate_randint_relations(0.01, (nx, ny), p0=0., maxvalue=1)
#
#    regret = SameEleNeigh(regdists, bool_input_idx=False)
#    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
#    regret._output_map = [m_out]
#    gret = RetrieverManager([ret0, ret1, regret])
##    regret = SameEleNeigh(regdists, bool_input_idx=False)
#
#    ## Features management
#    feat_arr0 = np.random.randint(0, 20, (n, 1))
#
#    features = ImplicitFeatures(feat_arr0)
#    reindices = np.vstack([np.random.permutation(n) for i in range(5)])
#    perturbation = PermutationPerturbation(reindices.T)
#
#    features.add_perturbations(perturbation)
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
#    feats_ret = FeaturesManager([features], descriptormodels=countdesc,
#                                maps_vals_i=map_vals_i)
##    feats_ret.add_aggregations((locs, griddisc), regret)
##
##    ## Descriptor
#    #avgdesc = AvgDescriptor()
#
#    ## External testing for looping
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

############# TO Input in exhaustive testing    
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
