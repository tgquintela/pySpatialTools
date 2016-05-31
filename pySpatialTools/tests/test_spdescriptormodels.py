
"""
test_spdescriptormodels
-----------------------
testing spatial descriptor models utilities.

"""

import numpy as np
import os
import copy
from itertools import product
import signal

## Retrieve
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import SameEleNeigh, KRetriever,\
    CircRetriever, RetrieverManager, WindowsRetriever
from pySpatialTools.Retrieve.retrievers import Retriever
from pySpatialTools.Retrieve.tools_retriever import create_aggretriever,\
    dummy_implicit_outretriver, dummy_explicit_outretriver,\
    avgregionlocs_outretriever

# Artificial data
from pySpatialTools.utils.artificial_data import generate_randint_relations

## Utilities
from pySpatialTools.utils.selectors import Sp_DescriptorSelector

## Features
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import\
    ImplicitFeatures, ExplicitFeatures, Features

from pySpatialTools.utils.perturbations import PermutationPerturbation
from pySpatialTools.utils.mapper_vals_i import create_mapper_vals_i

## Descriptormodel
from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor, NBinsHistogramDesc, SparseCounter
from pySpatialTools.FeatureManagement import SpatialDescriptorModel,\
    _spdesc_parsing_creation
from pySpatialTools.FeatureManagement.spatial_descriptormodels import\
    create_aggfeatures, _parse_aggregation_feat

#from ..utils.artificial_data import create_random_image
from ..utils.util_external import Logger
from ..io.io_images import create_locs_features_from_image


def test():
    n, nx, ny = 100, 100, 100
    m, rei = 3, 5
    locs = np.random.random((n, 2))*10
    ## Retrievers management
    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, .3, ifdistance=True)
    #countdesc = Countdescriptor()

    ## Other functions
    def map_indices(s, i):
        if s._pos_inputs is not None:
            return s._pos_inputs.start + s._pos_inputs.step*i
        else:
            return i

    def halting_f(signum, frame):
        raise Exception("Not error time.")

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
    def impossible_instantiation(selected, p, ret, feat):
        i_ret, sel, agg, pert = p
        p_ind, m_ind, n_desc, i_feat = selected
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

    def compulsary_instantiation_errors(selected, p, ret, feat):
        i_ret, sel, agg, pert = p
        p_ind, m_ind, n_desc, i_feat = selected
        checker = False

        ## Cases
        if p_ind == []:
            checker = True

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
        return checker

    def test_methods(methods, input_):
        """Test proper methods output for selectors indications."""
#        print methods, input_
        assert(len(methods) == 3)
        assert(methods[0] in [True, False])

        if methods[1] is None:
            assert(methods[2] is None)
        elif type(input_) == int:
            assert(type(methods[1]) == tuple)
            assert(type(methods[2]) == tuple)
            assert(len(methods[1]) == 2)
            assert(len(methods[2]) == 3)
            assert(all([len(e) == 2 for e in methods[2]]))
            assert(all([type(e) == tuple for e in methods[2]]))
        else:
            assert(type(input_) == list)
            assert(type(methods[1]) == list)
            assert(type(methods[2]) == list)
            assert(len(methods[1]) == len(input_))
            assert(len(methods[2]) == len(input_))
            for i in range(len(methods[1])):
                assert(type(methods[1][i]) == tuple)
                assert(type(methods[2][i]) == tuple)
                assert(len(methods[1][i]) == 2)
                assert(len(methods[2][i]) == 3)
                assert(all([len(e) == 2 for e in methods[2][i]]))
                assert(all([type(e) == tuple for e in methods[2][i]]))

    ###########################################################################
    ###########################################################################
    ######## Testing aggregations preparation
    ## Testing all possible aggregation_in
    agg_f_ret = None
    desc_in, desc_out = AvgDescriptor(), AvgDescriptor()
    feats = ImplicitFeatures(np.random.random((100, 10)),
                             characterizer=AvgDescriptor())

    agg_in = agg_f_ret, desc_in, {}, {}, desc_out
    res = _parse_aggregation_feat(agg_in, feats)
    assert(type(res) == tuple)
    assert(len(res) == 5)
    agg_in = agg_f_ret, desc_in, {}, {}
    res = _parse_aggregation_feat(agg_in, feats)
    assert(type(res) == tuple)
    assert(len(res) == 5)
    agg_in = agg_f_ret, {}, {}
    res = _parse_aggregation_feat(agg_in, feats)
    assert(type(res) == tuple)
    assert(len(res) == 5)
    agg_in = agg_f_ret, desc_in, desc_out
    res = _parse_aggregation_feat(agg_in, feats)
    assert(type(res) == tuple)
    assert(len(res) == 5)
    agg_in = (agg_f_ret, )
    res = _parse_aggregation_feat(agg_in, feats)
    assert(type(res) == tuple)
    assert(len(res) == 5)

    # Creation standard aggregation_info
    disc = GridSpatialDisc((5, 5), xlim=(0, 1), ylim=(0, 1))
    locs = np.random.random((100, 2))
    regs = disc.discretize(locs)
    disc_info = locs, regs, disc

    retriever_in = (KRetriever, {'info_ret': 4})
    retriever_out = (KRetriever, {'info_ret': 4})
    aggregating = avgregionlocs_outretriever, (avgregionlocs_outretriever, )

    aggregation_info = disc_info, retriever_in, retriever_out, aggregating
    # Creation of aggregation objects
    aggretriever = create_aggretriever(aggregation_info)
    assert(isinstance(aggretriever, Retriever))
    aggfeatures = create_aggfeatures(aggregation_info, feats)
    assert(isinstance(aggfeatures, Features))

    ###########################################################################
    ###########################################################################
    ######## Testing instantiation spdesc
    ## TODO: bool_input_idx=False

    # Aggregation
    disc = GridSpatialDisc((5, 5), xlim=(0, 1), ylim=(0, 1))
    retriever_in = (KRetriever, {'info_ret': 4})
    retriever_out = (KRetriever, {'info_ret': 4})
    aggregating = avgregionlocs_outretriever, (avgregionlocs_outretriever, )
    aggregation_info = disc, retriever_in, retriever_out, aggregating

    # Locs and retrievers
    n_in, n_out = 50, 50  # TODO: Different sizes and easy manage
    locs_input = np.random.random((n_in, 2))
    locs1 = np.random.random((n_out, 2))
    locs2 = np.random.random((n_out, 2))

    # Features
    aggfeats = np.random.random((n_out, m, rei))
    featsarr0 = np.random.random((n_out, m))
    featsarr1 = np.random.random((n_out, m))
    featsarr2 = np.vstack([np.random.randint(0, 10, n_out)
                           for i in range(m)]).T

    def new_retrievers_creation():
        ret0 = KRetriever(locs1, autolocs=locs_input, info_ret=3,
                          bool_input_idx=True)
        ret1 = [ret0, CircRetriever(locs2, info_ret=0.1, autolocs=locs_input,
                                    bool_input_idx=True)]
        ret2 = RetrieverManager(ret0)
        pos_rets = [ret0, ret1, ret2]
        return pos_rets

    pos_rets = range(3)

    ## Possible feats
    def new_features_creation():
        feats0 = ExplicitFeatures(aggfeats)
        feats1 = ImplicitFeatures(featsarr0)
        feats2 = FeaturesManager(ExplicitFeatures(aggfeats))

        pos_feats = [feats0, feats1, aggfeats, featsarr0, feats2]
        return pos_feats

    pos_feats = range(5)

    # Selectors
    arrayselector0 = np.zeros((n_in, 8))
    arrayselector1 = np.zeros((n_in, 2)), np.zeros((n_in, 6))
    arrayselector2 = np.zeros((n_in, 2)), tuple([np.zeros((n_in, 2))]*3)
    functselector0 = lambda idx: ((0, 0), ((0, 0), (0, 0), (0, 0)))
    functselector1 = lambda idx: (0, 0), lambda idx: ((0, 0), (0, 0), (0, 0))
    tupleselector0 = (0, 0), (0, 0, 0, 0, 0, 0)
    tupleselector1 = (0, 0, 0, 0, 0, 0, 0, 0)
    tupleselector2 = (0, 0), ((0, 0), (0, 0), (0, 0))

    listselector = None
    selobj = Sp_DescriptorSelector(*arrayselector1)
    pos_selectors = [None, arrayselector0, arrayselector1, arrayselector2,
                     functselector0, functselector1,
                     tupleselector0, tupleselector1, tupleselector2,
                     Sp_DescriptorSelector(arrayselector0)]
    pos_agg = [None]

    ## Perturbations
    reindices0 = np.arange(n_out)
    reindices = np.vstack([reindices0]+[np.random.permutation(n_out)
                                        for i in range(rei-1)]).T
    perturbation = PermutationPerturbation(reindices)
    pos_pert = [None, perturbation]

    ## Random exploration
    pos_loop_ind = [None, 20, (0, n_in, 1), slice(0, n_in, 1), []]
    pos_loop_mapin = [None, map_indices]
    pos_name_desc = [None, '', 'random_desc']
    # Possible feats
    # Random exploration possibilities
    pos_random = [pos_loop_ind, pos_loop_mapin, pos_name_desc, pos_feats]

    possibilities = [pos_rets, pos_selectors, pos_agg, pos_pert]

    s = 0
    for p in product(*possibilities):
        i_ret, sel, agg, pert = p
        ## Random exploration of parameters
        selected, indices = random_pos_space_exploration(pos_random)
        p_ind, m_ind, n_desc, i_feat = selected
        ## Classes renewal
        rets_cand = new_retrievers_creation()
        feats_cand = new_features_creation()
        # Retrievers
        ret = rets_cand[i_ret]
        feat = feats_cand[i_feat]

#        print indices
#        print p, selected
        ## Impossible cases
        checker1 = impossible_instantiation(selected, p, ret, feat)
        checker2 = compulsary_instantiation_errors(selected, p, ret, feat)
        if checker1 or checker2:
            continue
        ## Testing instantiation
        spdesc = SpatialDescriptorModel(retrievers=ret, featurers=feat,
                                        mapselector_spdescriptor=sel,
                                        pos_inputs=p_ind, map_indices=m_ind,
                                        perturbations=pert, aggregations=agg,
                                        name_desc=n_desc)
#        print s
        #### Function testing
        ## Auxiliar functions
        spdesc.add_perturbations(pert)
        spdesc.set_loop(p_ind, m_ind)
        spdesc._map_indices(spdesc, 0)
        for i in spdesc.iter_indices():
            methods = spdesc._get_methods(i)
            test_methods(methods, i)

        methods = spdesc._get_methods(0)
        test_methods(methods, 0)
        methods = spdesc._get_methods(10)
        test_methods(methods, 10)
        methods = spdesc._get_methods([0])
        test_methods(methods, [0])
        methods = spdesc._get_methods([0, 1, 2])
        test_methods(methods, [0, 1, 2])

        desc = spdesc._compute_descriptors(10)
        desc = spdesc._compute_descriptors([10])
        desc = spdesc._compute_descriptors([0, 1, 2])

        desc = spdesc.compute(10)
        desc = spdesc.compute([10])
        desc = spdesc.compute([0, 1, 2])

        #Retrieverdriven
        aux_i = 0
        for desc_i, vals_i in spdesc.compute_nets_i():
            assert(len(desc_i) == len(vals_i))
            assert(len(desc_i) == spdesc.featurers.k_perturb+1)
            aux_i += 1
            if aux_i == 100:
                break
        aux_i = 0
        for desc_ik, vals_ik in spdesc.compute_net_ik():
            aux_i += 1
            if aux_i == 100:
                break

        ## Loops
#        for idx in spdesc.iter_indices():
#            break
#        for vals_ik, desc_ik in spdesc.compute_net_ik():
#            #assert(vals_ik)
#            #assert(desc_ik)
#            break
#        for desc_i, vals_i in spdesc.compute_net_i():
#            #assert(vals_ik)
#            #assert(desc_ik)
#            break

        ## Global computations
#        try:
#            signal.signal(signal.SIGALRM, halting_f)
#            signal.alarm(0.01)   # 0.01 seconds
#            spdesc.compute()
#        except Exception as e:
#            logi = e == "Not error time."
#            if not logi:
#                spdesc.compute()
#        try:
#            signal.signal(signal.SIGALRM, halting_f)
#            signal.alarm(0.01)   # 0.01 seconds
#            spdesc._compute_nets()
#        except Exception as e:
#            logi = e == "Not error time."
#            if not logi:
#                spdesc._compute_nets()
#        try:
#            ## Testing compute_retdriven
#            signal.signal(signal.SIGALRM, halting_f)
#            signal.alarm(0.01)   # 0.01 seconds
#            spdesc._compute_retdriven()
#        except Exception as e:
#            logi = e == "Not error time."
#            if not logi:
#                spdesc._compute_retdriven()
#        try:
#            logfile = Logger('logfile.log')
#            signal.signal(signal.SIGALRM, halting_f)
#            signal.alarm(0.01)   # 0.01 seconds
#            spdesc.compute_process(logfile, lim_rows=100000, n_procs=0)
#            os.remove('logfile.log')
#        except Exception as e:
#            os.remove('logfile.log')
#            logi = e == "Not error time."
#            if not logi:
#                spdesc.compute_process(logfile, lim_rows=100000, n_procs=0)

        ## Testing aggregations
        if len(spdesc.retrievers) == len(spdesc.featurers):
            spdesc.add_aggregations(aggregation_info)
        else:
            spdesc.add_aggregations(aggregation_info, ([0], [0]))
        s += 1

    feats1 = ImplicitFeatures(featsarr0)

    m_vals_i = np.random.randint(0, 5, 50)
    ret = CircRetriever(locs1, autolocs=locs_input, info_ret=3,
                        bool_input_idx=True)
    feat = FeaturesManager(feats1, maps_vals_i=m_vals_i, mode='sequential',
                           descriptormodels=None)
    spdesc = SpatialDescriptorModel(retrievers=ret, featurers=feat,
                                    mapselector_spdescriptor=None,
                                    perturbations=perturbation,
                                    aggregations=None, name_desc=n_desc)
    ## Complete processes
    spdesc.compute()
    logfile = Logger('logfile.log')
    spdesc.compute_process(logfile, lim_rows=100000, n_procs=0)
    os.remove('logfile.log')
    spdesc._compute_nets()
    spdesc._compute_retdriven()

    ############
    ### Auxiliar functions
    ####
    spdesc = _spdesc_parsing_creation(ret, feat)
    assert(isinstance(spdesc, SpatialDescriptorModel))
    res = create_aggfeatures(spdesc, None)
    assert(isinstance(res, ExplicitFeatures))

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
