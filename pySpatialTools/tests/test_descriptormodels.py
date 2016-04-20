
"""
test_spdescriptormodels
-----------------------
testing descriptor models utilities.

"""

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
    AvgDescriptor, PjensenDescriptor, SumDescriptor, NBinsHistogramDesc,\
    SparseCounter
from pySpatialTools.FeatureManagement.descriptormodel import\
    GeneralDescriptor
from pySpatialTools.FeatureManagement.aux_descriptormodels import *

from pySpatialTools.FeatureManagement import SpatialDescriptorModel

import numpy as np

## Invocable characterizer functions
from pySpatialTools.FeatureManagement.aux_descriptormodels import\
    characterizer_1sh_counter, characterizer_summer, characterizer_average,\
    sum_reducer, avg_reducer, sum_addresult_function,\
    append_addresult_function, replacelist_addresult_function,\
    null_completer, weighted_completer, sparse_dict_completer,\
    aggregator_1sh_counter, aggregator_summer, aggregator_average,\
    counter_featurenames, array_featurenames,\
    count_out_formatter_general, null_out_formatter,\
    count_out_formatter_dict2array


def test():
    n = 100
    locs = np.random.random((n, 2))*100
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))

    ########################### Auxdescriptormodels ###########################
    ###########################################################################
    #################################
    #### Reducer testing
    def creation_agg(listfeats):
        n_iss = np.random.randint(1, 10)
        if listfeats:
            aggdesc = []
            for i in range(n_iss):
                keys = np.unique(np.random.randint(0, 20, 10))
                values = np.random.random(len(keys))
                aggdesc.append(dict(zip(keys, values)))
        else:
            n_feats = 20
            aggdesc = np.random.random((n_iss, n_feats))
        p_aggpos = None
        return aggdesc, p_aggpos

    ## Reducer
    aggdesc, p_aggpos = creation_agg(True)
    sum_reducer(aggdesc, p_aggpos)
    avg_reducer(aggdesc, p_aggpos)
    aggdesc, p_aggpos = creation_agg(False)
    sum_reducer(aggdesc, p_aggpos)
    avg_reducer(aggdesc, p_aggpos)
    aggdesc, p_aggpos = creation_agg(False)
    sum_reducer(list(aggdesc), p_aggpos)
    avg_reducer(list(aggdesc), p_aggpos)

    #################################
    #### Outformatters
    def creation_outformatter():
        n_iss = np.random.randint(1, 10)
        outfeats = [str(e) for e in np.arange(20)]
        feats = []
        for i in range(n_iss):
            keys = np.unique(np.random.randint(0, 20, 10))
            values = np.random.random(len(keys))
            feats.append(dict(zip(keys, values)))
        return feats, outfeats
    _out = ['ndarray', 'dict']
    feats, outfeats = creation_outformatter()
    count_out_formatter_general(feats, outfeats, _out[0], 0)
    count_out_formatter_general(feats, outfeats, _out[1], 0)
    null_out_formatter(feats, outfeats, _out[0], 0)
    try:
        boolean = False
        count_out_formatter_general(feats, outfeats, '', 0)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        array_feats = np.random.random((10, 1))
        count_out_formatter_general(array_feats, outfeats, _out[1], 0)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    #################################
    #### Featurenames
    def creation_features(listfeats):
        n_iss = np.random.randint(1, 10)
        if listfeats:
            feats = []
            for i in range(n_iss):
                keys = np.unique(np.random.randint(0, 20, 10))
                values = np.random.random(len(keys))
                feats.append(dict(zip(keys, values)))
        else:
            feats = np.random.randint(0, 20, n_iss).reshape((n_iss, 1))
        return feats
    # List feats
    features_o = creation_features(True)
    counter_featurenames(features_o)
    list_featurenames(features_o)
    try:
        boolean = False
        array_featurenames(features_o)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    # Array feats
    features_o = creation_features(False)
    counter_featurenames(features_o)
    array_featurenames(features_o)

    try:
        boolean = False
        list_featurenames(features_o)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        counter_featurenames(None)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        array_featurenames(None)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    #################################
    #### Characterizers
    # TODO: listdicts feats based characterizers
    def creation_feats2characterize(listfeats):
        n_iss = np.random.randint(1, 10)
        if listfeats:
            pass
        else:
            pointfeats = np.random.randint(0, 20, n_iss).reshape((n_iss, 1))
        point_pos = None
        return pointfeats, point_pos

    pointfeats, point_pos = creation_feats2characterize(False)
    characterizer_1sh_counter(pointfeats, point_pos)
    characterizer_summer(pointfeats, point_pos)
    characterizer_average(pointfeats, point_pos)

    characterizer_1sh_counter(list(pointfeats), point_pos)
    characterizer_summer(list(pointfeats), point_pos)
    characterizer_average(list(pointfeats), point_pos)

    #################################
    #### Characterizers
    # TODO: listdicts feats based characterizers
    aggregator_1sh_counter(pointfeats, point_pos)
    aggregator_summer(pointfeats, point_pos)
    aggregator_average(pointfeats, point_pos)

    #################################
    #### add2results
    def creation_x_i(listfeats, n_k, n_iss, n_feats):
        if listfeats:
            x_i = []
            for k in range(n_k):
                x_i_k = []
                for i in range(n_iss):
                    keys = np.unique(np.random.randint(0, n_feats, n_feats))
                    keys = [str(e) for e in keys]
                    values = np.random.random(len(keys))
                    x_i_k.append(dict(zip(keys, values)))
                x_i.append(x_i_k)
        else:
            x_i = np.random.random((n_k, n_iss, n_feats))
        return x_i

    def creation_add2res(type_):
        ## Preparations
        n_feats = np.random.randint(1, 20)
        n_k = np.random.randint(1, 20)
        n_iss = np.random.randint(1, 20)
        max_vals_i = np.random.randint(1, 20)
        vals_i = []
        for i in range(n_k):
            vals_i.append(np.random.randint(0, max_vals_i, n_iss))
        if type_ == 'replacelist':
            x = [[[], []]]*n_k
            x_i = creation_x_i(True, n_k, n_iss, n_feats)
        elif type_ == 'append':
            x = [[[]]*n_iss]*n_k
            x_i = creation_x_i(True, n_k, n_iss, n_feats)
        elif type_ == 'sum':
            x_i = creation_x_i(False, n_k, n_iss, n_feats)
            x = np.random.random((max_vals_i, n_feats, n_k))
        return x, x_i, vals_i

    types = ['replacelist', 'append', 'sum']
#    x, x_i, vals_i = creation_add2res(types[0])
#    measure_spdict_unknown = replacelist_addresult_function(x, x_i, vals_i)
#    x, x_i, vals_i = creation_add2res(types[1])
#    measure_spdict_known = append_addresult_function(x, x_i, vals_i)
    x, x_i, vals_i = creation_add2res(types[2])
    measure_array = sum_addresult_function(x, x_i, vals_i)

    #################################
    #### Completers
#    sparse_dict_completer(measure_spdict_known)
#    sparse_dict_completer_unknown(measure_spdict_unknown)
    null_completer(measure_array)
    global_info = np.random.random(len(measure_array))
    weighted_completer(measure_array, global_info)
    global_info = np.random.random(measure_array.shape)
    weighted_completer(measure_array, global_info)

    ############################# Descriptormodels ############################
    ###########################################################################
    #################################
    #### SumDescriptor
    sumdesc = SumDescriptor()
    characs = np.random.random((10, 5))
    point_pos = np.random.random((10, 5))
    measure = np.random.random((100, 10, 2))

    sumdesc.compute_characs(characs, point_pos)
    sumdesc.compute_characs(characs, None)

    sumdesc.reducer(characs, point_pos)
    sumdesc.reducer(characs, None)

    sumdesc.aggdescriptor(characs, point_pos)
    sumdesc.aggdescriptor(characs, None)

    sumdesc.to_complete_measure(measure)
    #sumdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)

    # Not specific
    sumdesc.set_global_info(None)
    sumdesc.set_functions(None, None)

    #################################
    #### AvgDescriptor
    avgdesc = AvgDescriptor()
    characs = np.random.random((10, 5))
    point_pos = np.random.random((10, 5))
    measure = np.random.random((100, 10, 2))

    avgdesc.compute_characs(characs, point_pos)
    avgdesc.compute_characs(characs, None)

    avgdesc.reducer(characs, point_pos)
    avgdesc.reducer(characs, None)

    avgdesc.aggdescriptor(characs, point_pos)
    avgdesc.aggdescriptor(characs, None)

    avgdesc.to_complete_measure(measure)
    #avgdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)

    # Not specific
    avgdesc.set_global_info(None)
    avgdesc.set_functions(None, None)

    #################################
    #### Countdescriptor
    countdesc = Countdescriptor()
    characs = np.random.randint(0, 10, 50).reshape((10, 5))
    point_pos = np.random.random((10, 5))
    measure = np.random.random((100, 10, 2))

    countdesc.compute_characs(characs, point_pos)
    countdesc.compute_characs(characs, None)

    countdesc.reducer(characs, point_pos)
    countdesc.reducer(characs, None)

    countdesc.aggdescriptor(characs, point_pos)
    countdesc.aggdescriptor(characs, None)

    countdesc.to_complete_measure(measure)
    #countdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)

    # Not specific
    countdesc.set_global_info(None)
    countdesc._format_default_functions()
    countdesc.set_functions(None, None)
    countdesc.set_functions(None, 'dict')

    #################################
    #### Pjensen
    pjensen = PjensenDescriptor()
    # Specific
    features = list(np.arange(20)) + list(np.random.randint(0, 20, 80))
    features = np.array(features).reshape((100, 1))
    pjensen.set_global_info(features)

    pjensen = PjensenDescriptor(features)
    characs = np.random.randint(0, 10, 50).reshape((10, 5))
    point_pos = np.random.random((10, 5))
    measure = np.random.randint(0, 50, 20*20).reshape((20, 20, 1))

    # Functions
    pjensen.compute_characs(characs, point_pos)
    pjensen.compute_characs(characs, None)

    pjensen.reducer(characs, point_pos)
    pjensen.reducer(characs, None)

    pjensen.aggdescriptor(characs, point_pos)
    pjensen.aggdescriptor(characs, None)

    pjensen.to_complete_measure(measure)
    #pjensen.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)

    # Not specific
    pjensen._format_default_functions()
    pjensen.set_functions(None, None)
    pjensen.set_functions(None, 'dict')

    #################################
    #### SparseCounter
#    spcountdesc = SparseCounter()
#
#    spcountdesc.compute_characs(characs, point_pos)
#    spcountdesc.compute_characs(characs, None)
#
#    spcountdesc.reducer(characs, point_pos)
#    spcountdesc.reducer(characs, None)
#
#    spcountdesc.aggdescriptor(characs, point_pos)
#    spcountdesc.aggdescriptor(characs, None)
#
#    spcountdesc.to_complete_measure(measure)
#    #spcountdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)
#
#    # Not specific
#    spcountdesc.set_global_info(None)
#    spcountdesc.set_functions(None, None)

    #################################
    #### NBinsHistogramDesc
    nbinsdesc = NBinsHistogramDesc(5)
    characs = np.random.randint(0, 10, 50).reshape((10, 5))
    point_pos = np.random.random((10, 5))
    measure = np.random.random((100, 10, 2))

    nbinsdesc.compute_characs(characs, point_pos)
    nbinsdesc.compute_characs([characs], None)

    nbinsdesc.reducer(characs, point_pos)
    nbinsdesc.reducer(characs, None)

    nbinsdesc.aggdescriptor(characs, point_pos)
    nbinsdesc.aggdescriptor(characs, None)

    nbinsdesc.to_complete_measure(measure)
    #nbinsdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)

    # Specific
    nbinsdesc._format_default_functions()
    nbinsdesc.set_functions(None, None)
    nbinsdesc.set_functions(None, 'dict')
    features = np.random.random((100, 5))
    nbinsdesc.set_global_info(features, True)
    nbinsdesc.set_global_info(features, False)
    # Not specific


#    ret0 = KRetriever(locs, 3, ifdistance=True)
#    ret1 = CircRetriever(locs, 3, ifdistance=True)
#    gret0 = RetrieverManager([ret0])
#    gret1 = RetrieverManager([ret1])
#
#    ## Create MAP VALS (indices)
#    corr_arr = -1*np.ones(n).astype(int)
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
#    feats0 = ImplicitFeatures(feat_arr0)
#    feats1 = ImplicitFeatures(feat_arr1)
#
#    avgdesc = AvgDescriptor()
#    countdesc = Countdescriptor()
#    pjensendesc = PjensenDescriptor()
#
#    feats_ret0 = FeaturesManager(feats0, countdesc, maps_vals_i=map_vals_i)
#    feats_ret1 = FeaturesManager([feats1], avgdesc, maps_vals_i=map_vals_i)
#    feats_ret2 = FeaturesManager(feats0, pjensendesc, maps_vals_i=map_vals_i)
#
#    sp_model0 = SpatialDescriptorModel(gret0, feats_ret1)
#    sp_model1 = SpatialDescriptorModel(gret1, feats_ret1)
#    sp_model2 = SpatialDescriptorModel(gret0, feats_ret0)
#    sp_model3 = SpatialDescriptorModel(gret1, feats_ret0)
#    sp_model4 = SpatialDescriptorModel(gret0, feats_ret2)
#    sp_model5 = SpatialDescriptorModel(gret1, feats_ret2)
#
#    corr = sp_model0.compute()
#    corr = sp_model1.compute()
#    corr = sp_model2.compute()
#    corr = sp_model3.compute()
#    corr = sp_model4.compute()
#    corr = sp_model5.compute()
#
#    ### Testing auxiliar descriptormodels functions
#    # Artificial data
#    contfeats, point_pos = np.random.random(5), np.random.random(5)
#    catfeats = np.random.randint(0, 10, 5)
#    aggdescriptors_idxs = np.random.random((10, 5))
#    x, x_i, vals_i = np.zeros((1, 1, 1)), np.zeros((1, 1)), [[0]]
#    # Characterizers
#    characterizer_1sh_counter(catfeats, point_pos)
#    characterizer_summer(contfeats, point_pos)
#    characterizer_average(contfeats, point_pos)
#    # Reducers
#    sum_reducer([aggdescriptors_idxs], point_pos)
#    sum_reducer([{9: 0, 8: 1, 4: 7, 3: 0, 1: 0}], point_pos)
#    avg_reducer(aggdescriptors_idxs, point_pos)
#    avg_reducer([{9: 0, 8: 1, 4: 7, 3: 0, 1: 0}], point_pos)
#
#    # Add2result
#    sum_addresult_function(x, x_i, vals_i)
#    append_addresult_function([[[]]], x_i, vals_i)
#    replacelist_addresult_function([[[], []]], x_i, vals_i)
#    # Completers
#    null_completer(np.array([1]))
#    weighted_completer(np.array([1]), np.array([1]))
#    weighted_completer(np.array([1]), None)
#    sparse_dict_completer([[[{0: 2}]]])
#    sparse_dict_completer([[[{0: 2}, {1: 3}]]])
#    # Aggregators
#    aggregator_1sh_counter(catfeats, point_pos)
#    aggregator_summer(catfeats, point_pos)
#    aggregator_average(catfeats, point_pos)
#    # Featurenames
#    counter_featurenames(np.random.randint(0, 10, 10).reshape((10, 1)))
#    try:
#        counter_featurenames([np.random.randint(0, 10, 10).reshape((10, 1))])
#        raise Exception
#    except:
#        pass
#    array_featurenames([np.random.random((10, 5))])
#    try:
#        array_featurenames(None)
#        raise Exception
#    except:
#        pass
#    # Out formatter
#    count_out_formatter_general(catfeats, catfeats, 'dict', 0)
#    try:
#        count_out_formatter_general(catfeats, catfeats[:3], 'dict', 0)
#        raise Exception
#    except:
#        pass
#    null_out_formatter(catfeats, catfeats, 'dict', 0)
#
#    ### Testing descriptors
#    # Artificial data
#    contfeats, point_pos = np.random.random(5), np.random.random(5)
#    catfeats = np.random.randint(0, 10, 5)
#    aggdescriptors_idxs = np.random.random((10, 5))
#
#    # Descriptors
#    avgdesc = AvgDescriptor()
#    countdesc = Countdescriptor()
#    pjensendesc = PjensenDescriptor()
#    sumdesc = SumDescriptor()
#    nbinsdesc = NBinsHistogramDesc(5)
#    sparsedesc = SparseCounter()
#
#    avgdesc.compute_characs(contfeats, point_pos)
#    avgdesc.reducer(aggdescriptors_idxs, point_pos)
#    avgdesc.aggdescriptor(contfeats, point_pos)
#    countdesc.compute_characs(catfeats, point_pos)
#    countdesc.reducer(aggdescriptors_idxs, point_pos)
#    countdesc.aggdescriptor(catfeats, point_pos)
#    pjensendesc.compute_characs(catfeats, point_pos)
#    pjensendesc.reducer(aggdescriptors_idxs, point_pos)
#    pjensendesc.aggdescriptor(catfeats, point_pos)
#    sumdesc.compute_characs(contfeats, point_pos)
#    sumdesc.reducer(aggdescriptors_idxs, point_pos)
#    sumdesc.aggdescriptor(contfeats, point_pos)
#    nbinsdesc.compute_characs(contfeats, point_pos)
#    nbinsdesc.reducer(aggdescriptors_idxs, point_pos)
#    nbinsdesc.aggdescriptor(contfeats, point_pos)
#    sparsedesc.compute_characs(catfeats, point_pos)
#    sparsedesc.reducer(aggdescriptors_idxs, point_pos)
#    sparsedesc.aggdescriptor(catfeats, point_pos)
#
#    ## GeneralDescriptor
#    GeneralDescriptor(characterizer_summer, sum_reducer, characterizer_summer,
#                      null_completer)
