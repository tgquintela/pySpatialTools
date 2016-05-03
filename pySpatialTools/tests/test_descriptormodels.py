
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
from pySpatialTools.utils.artificial_data import\
    continuous_array_features, categorical_agg_dict_features,\
    categorical_array_features, continuous_dict_features,\
    categorical_dict_features, continuous_agg_array_features,\
    categorical_agg_array_features, continuous_agg_dict_features
from pySpatialTools.utils.artificial_data.artificial_measure import *

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
from pySpatialTools.FeatureManagement.aux_descriptormodels import *


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

    nnei, n_feats = np.random.randint(1, 1000), np.random.randint(1, 20)
    n_feats2 = [np.random.randint(1, 20) for i in range(n_feats)]
    n_iss = np.random.randint(1, 20)
    point_pos = [None]*n_iss

    ### Tests
    # Example objects

    pointfeats_arrayarray0 = [continuous_array_features(nnei, n_feats)]*n_iss
    pointfeats_listarray0 = np.array(pointfeats_arrayarray0)
    pointfeats_arrayarray1 = [categorical_array_features(nnei, n_feats)]*n_iss
    pointfeats_listarray1 = np.array(pointfeats_arrayarray1)
    pointfeats_arrayarray2 = [categorical_array_features(nnei, n_feats2)]*n_iss
    pointfeats_listarray2 = np.array(pointfeats_arrayarray2)
    pointfeats_listdict0 = [continuous_dict_features(nnei, n_feats)]*n_iss
    pointfeats_listdict1 = [categorical_dict_features(nnei, n_feats)]*n_iss

#    pointfeats_arrayarray0 = continuous_agg_array_features(n, n_feats, ks)
#    pointfeats_listarray0 = list(pointfeats_arrayarray0)
#    pointfeats_arrayarray1 = categorical_agg_array_features(n, n_feats, ks)
#    pointfeats_listarray1 = list(pointfeats_arrayarray1)
#    pointfeats_arrayarray2 = categorical_agg_array_features(n, n_feats2, ks)
#    pointfeats_listarray2 = list(pointfeats_arrayarray2)
#    pointfeats_listdict0 = continuous_agg_dict_features(n, n_feats, ks)
#    pointfeats_listdict1 = categorical_agg_dict_features(n, n_feats, ks)

    #################################
    #### Reducer
    ###############

    desc = sum_reducer(pointfeats_arrayarray0, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = sum_reducer(pointfeats_listarray0, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = sum_reducer(pointfeats_arrayarray1, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = sum_reducer(pointfeats_listarray1, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = sum_reducer(pointfeats_arrayarray2, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = sum_reducer(pointfeats_listarray2, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = sum_reducer(pointfeats_listdict0, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == dict)
    desc = sum_reducer(pointfeats_listdict1, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == dict)

    desc = avg_reducer(pointfeats_arrayarray0, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = avg_reducer(pointfeats_listarray0, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = avg_reducer(pointfeats_arrayarray1, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = avg_reducer(pointfeats_listarray1, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = avg_reducer(pointfeats_arrayarray2, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = avg_reducer(pointfeats_listarray2, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == np.ndarray)
    desc = avg_reducer(pointfeats_listdict0, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == dict)
    desc = avg_reducer(pointfeats_listdict1, point_pos)
    assert(type(desc) == list)
    assert(type(desc[0]) == dict)

#    aggdesc, p_aggpos = creation_agg(True)
#    sum_reducer(aggdesc, p_aggpos)
#    avg_reducer(aggdesc, p_aggpos)
#    aggdesc, p_aggpos = creation_agg(False)
#    sum_reducer(aggdesc, p_aggpos)
#    avg_reducer(aggdesc, p_aggpos)
#    aggdesc, p_aggpos = creation_agg(False)
#    sum_reducer(list(aggdesc), p_aggpos)
#    avg_reducer(list(aggdesc), p_aggpos)

    #################################
    #### Outformatters
    ###################
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
    #################
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
    ###################
    # We need 2nd level features so we use aggregation ones
    # [iss][nei]{feats} or [iss](nei, feats) or (iss, nei, feats)

    point_pos = None
    n, n_feats = np.random.randint(10, 1000), np.random.randint(1, 20)
    n_feats2 = [np.random.randint(1, 20) for i in range(n_feats)]
    ks = np.random.randint(1, 20)

    ### Tests
    # Example objects
    pointfeats_arrayarray0 = continuous_agg_array_features(n, n_feats, ks)
    pointfeats_listarray0 = list(pointfeats_arrayarray0)
    pointfeats_arrayarray1 = categorical_agg_array_features(n, n_feats, ks)
    pointfeats_listarray1 = list(pointfeats_arrayarray1)
    pointfeats_arrayarray2 = categorical_agg_array_features(n, n_feats2, ks)
    pointfeats_listarray2 = list(pointfeats_arrayarray2)
    pointfeats_listdict0 = continuous_agg_dict_features(n, n_feats, ks)
    pointfeats_listdict1 = categorical_agg_dict_features(n, n_feats, ks)

    # Counter
    characterizer_1sh_counter(pointfeats_arrayarray0, point_pos)
    characterizer_1sh_counter(pointfeats_arrayarray1, point_pos)
    characterizer_1sh_counter(pointfeats_listarray0, point_pos)
    characterizer_1sh_counter(pointfeats_listarray1, point_pos)

    # Summer
    characterizer_summer(pointfeats_arrayarray0, point_pos)
    characterizer_summer(pointfeats_listarray0, point_pos)
    characterizer_summer(pointfeats_arrayarray1, point_pos)
    characterizer_summer(pointfeats_listarray1, point_pos)
    characterizer_summer(pointfeats_arrayarray2, point_pos)
    characterizer_summer(pointfeats_listarray2, point_pos)
    characterizer_summer(pointfeats_listdict0, point_pos)
    characterizer_summer(pointfeats_listdict1, point_pos)

    characterizer_summer_array(pointfeats_arrayarray0, point_pos)
    characterizer_summer_array(pointfeats_listarray0, point_pos)
    characterizer_summer_array(pointfeats_arrayarray1, point_pos)
    characterizer_summer_array(pointfeats_listarray1, point_pos)
    characterizer_summer_array(pointfeats_arrayarray2, point_pos)
    characterizer_summer_array(pointfeats_listarray2, point_pos)

    characterizer_summer_listdict(pointfeats_listdict0, point_pos)
    characterizer_summer_listdict(pointfeats_listdict1, point_pos)

    characterizer_summer_listarray(pointfeats_listarray0, point_pos)
    characterizer_summer_listarray(pointfeats_listarray1, point_pos)
    characterizer_summer_listarray(pointfeats_listarray2, point_pos)
    characterizer_summer_arrayarray(pointfeats_arrayarray0, point_pos)
    characterizer_summer_arrayarray(pointfeats_arrayarray1, point_pos)
    characterizer_summer_arrayarray(pointfeats_arrayarray2, point_pos)

    # Average
    characterizer_average(pointfeats_arrayarray0, point_pos)
    characterizer_average(pointfeats_listarray0, point_pos)
    characterizer_average(pointfeats_arrayarray1, point_pos)
    characterizer_average(pointfeats_listarray1, point_pos)
    characterizer_average(pointfeats_arrayarray2, point_pos)
    characterizer_average(pointfeats_listarray2, point_pos)
    characterizer_average(pointfeats_listdict0, point_pos)
    characterizer_average(pointfeats_listdict1, point_pos)

    characterizer_average_array(pointfeats_arrayarray0, point_pos)
    characterizer_average_array(pointfeats_listarray0, point_pos)
    characterizer_average_array(pointfeats_arrayarray1, point_pos)
    characterizer_average_array(pointfeats_listarray1, point_pos)
    characterizer_average_array(pointfeats_arrayarray2, point_pos)
    characterizer_average_array(pointfeats_listarray2, point_pos)

    characterizer_average_listdict(pointfeats_listdict0, point_pos)
    characterizer_average_listdict(pointfeats_listdict1, point_pos)

    characterizer_average_listarray(pointfeats_listarray0, point_pos)
    characterizer_average_listarray(pointfeats_listarray1, point_pos)
    characterizer_average_listarray(pointfeats_listarray2, point_pos)
    characterizer_average_arrayarray(pointfeats_arrayarray0, point_pos)
    characterizer_average_arrayarray(pointfeats_arrayarray1, point_pos)
    characterizer_average_arrayarray(pointfeats_arrayarray2, point_pos)

    ## Testing utils
    f = characterizer_from_unitcharacterizer(lambda x, y: x[0])
    f(pointfeats_arrayarray0, [point_pos]*n)
    f(pointfeats_listarray0, [point_pos]*n)
    f(pointfeats_arrayarray1, [point_pos]*n)
    f(pointfeats_listarray1, [point_pos]*n)
    f(pointfeats_arrayarray2, [point_pos]*n)
    f(pointfeats_listarray2, [point_pos]*n)
    f(pointfeats_listdict0, [point_pos]*n)
    f(pointfeats_listdict1, [point_pos]*n)

    #################################
    #### Characterizers
    ###################

    # TODO: listdicts feats based characterizers

#    aggregator_1sh_counter(pointfeats, point_pos)
#    aggregator_summer(pointfeats, point_pos)
#    aggregator_average(pointfeats, point_pos)

    #################################
    #### add2results
#    def creation_x_i(listfeats, n_k, n_iss, n_feats):
#        if listfeats:
#            x_i = []
#            for k in range(n_k):
#                x_i_k = []
#                for i in range(n_iss):
#                    keys = np.unique(np.random.randint(0, n_feats, n_feats))
#                    keys = [str(e) for e in keys]
#                    values = np.random.random(len(keys))
#                    x_i_k.append(dict(zip(keys, values)))
#                x_i.append(x_i_k)
#        else:
#            x_i = np.random.random((n_k, n_iss, n_feats))
#        return x_i
#
#    def creation_add2res(type_):
#        ## Preparations
#        n_feats = np.random.randint(1, 20)
#        n_k = np.random.randint(1, 20)
#        n_iss = np.random.randint(1, 20)
#        max_vals_i = np.random.randint(1, 20)
#        vals_i = []
#        for i in range(n_k):
#            vals_i.append(np.random.randint(0, max_vals_i, n_iss))
#        if type_ == 'replacelist':
#            x = [[[], []]]*n_k
#            x_i = creation_x_i(True, n_k, n_iss, n_feats)
#        elif type_ == 'append':
#            x = [[[]]*n_iss]*n_k
#            x_i = creation_x_i(True, n_k, n_iss, n_feats)
#        elif type_ == 'sum':
#            x_i = creation_x_i(False, n_k, n_iss, n_feats)
#            x = np.random.random((max_vals_i, n_feats, n_k))
#        return x, x_i, vals_i
#
#    types = ['replacelist', 'append', 'sum']
#    x, x_i, vals_i = creation_add2res(types[0])
#    x, x_i, vals_i = creation_add2res(types[2])
#    x, x_i, vals_i = creation_add2res(types[1])

    n_feats = np.random.randint(2, 20)
    ks = np.random.randint(1, 20)
    n_iss = np.random.randint(1, 20)
    n_vals_i = np.random.randint(2, 20)

    vals_i = create_vals_i(n_iss, n_vals_i, ks)

    x = create_artificial_measure_replacelist(ks, n_vals_i, n_feats)
    x_i = create_empty_features_dict(n_feats, n_iss, ks)
    measure_spdict_unknown = replacelist_addresult_function(x, x_i, vals_i)
    x = create_artificial_measure_replacelist(ks, n_vals_i, n_feats, True)
    measure_spdict_unknown = replacelist_addresult_function(x, x_i, vals_i)

    x = create_artificial_measure_append(ks, n_vals_i, n_feats)
    append_addresult_function(x, x_i, vals_i)
    x[0][0] = x[0][0][0]
    append_addresult_function(x, x_i, vals_i)

    x = create_artificial_measure_array(ks, n_vals_i, n_feats)
    x_i = create_empty_features_array(n_feats, n_iss, ks)

    measure_array = sum_addresult_function(x, x_i, vals_i)

    #################################
    #### Completers
    x = create_artificial_measure_append(ks, n_vals_i, n_feats)
    sparse_dict_completer(x)
    sparse_dict_completer_unknown(measure_spdict_unknown)
    null_completer(measure_array)
    global_info = np.random.random(len(measure_array))
    weighted_completer(measure_array, global_info)
    global_info = np.random.random(measure_array.shape)
    weighted_completer(measure_array, global_info)
    weighted_completer(measure_array, None)

    ############################# Descriptormodels ############################
    ###########################################################################
    #################################
    #### SumDescriptor
    point_pos = None
    measure = np.random.random((100, 10, 2))
#    characs = np.random.random((10, 5))
#    feats = continuous_array_features(100, 10)
#    feats_dict = continuous_dict_features(100, 10)

    sumdesc = SumDescriptor()
    sumdesc.compute_characs(pointfeats_arrayarray0, point_pos)
    sumdesc.compute_characs(pointfeats_listarray0, point_pos)
    sumdesc.compute_characs(pointfeats_arrayarray1, point_pos)
    sumdesc.compute_characs(pointfeats_listarray1, point_pos)
    sumdesc.compute_characs(pointfeats_arrayarray2, point_pos)
    sumdesc.compute_characs(pointfeats_listarray2, point_pos)
    sumdesc.compute_characs(pointfeats_listdict0, point_pos)
    sumdesc.compute_characs(pointfeats_listdict1, point_pos)

    sumdesc = SumDescriptor('array')
    sumdesc.compute_characs(pointfeats_arrayarray0, point_pos)
    sumdesc.compute_characs(pointfeats_listarray0, point_pos)
    sumdesc.compute_characs(pointfeats_arrayarray1, point_pos)
    sumdesc.compute_characs(pointfeats_listarray1, point_pos)
    sumdesc.compute_characs(pointfeats_arrayarray2, point_pos)
    sumdesc.compute_characs(pointfeats_listarray2, point_pos)

    sumdesc = SumDescriptor('listdict')
    sumdesc.compute_characs(pointfeats_listdict0, point_pos)
    sumdesc.compute_characs(pointfeats_listdict1, point_pos)

    sumdesc = SumDescriptor('listarray')
    sumdesc.compute_characs(pointfeats_listarray0, point_pos)
    sumdesc.compute_characs(pointfeats_listarray1, point_pos)
    sumdesc.compute_characs(pointfeats_listarray2, point_pos)

    sumdesc = SumDescriptor('arrayarray')
    sumdesc.compute_characs(pointfeats_arrayarray0, point_pos)
    sumdesc.compute_characs(pointfeats_arrayarray1, point_pos)
    sumdesc.compute_characs(pointfeats_arrayarray2, point_pos)

#    sumdesc.compute_characs(feats_dict, point_pos)
#    sumdesc.compute_characs(feats_dict, None)
#    sumdesc.reducer(feats_dict, point_pos)
#    sumdesc.reducer(feats_dict, None)
#    sumdesc.aggdescriptor(feats_dict, point_pos)
#    sumdesc.aggdescriptor(feats_dict, None)

    # Not specific
    sumdesc.to_complete_measure(measure)
    #sumdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)
    sumdesc.set_global_info(None)
    sumdesc.set_functions(None, None)

    #################################
    #### AvgDescriptor
    point_pos = np.random.random((10, 5))
    measure = np.random.random((100, 10, 2))
#    avgdesc = AvgDescriptor()
#    characs = np.random.random((10, 5))
#    feats = continuous_array_features(100, 10)
#    feats_dict = continuous_dict_features(100, 10)

    avgdesc = AvgDescriptor()
    avgdesc.compute_characs(pointfeats_arrayarray0, point_pos)
    avgdesc.compute_characs(pointfeats_listarray0, point_pos)
    avgdesc.compute_characs(pointfeats_arrayarray1, point_pos)
    avgdesc.compute_characs(pointfeats_listarray1, point_pos)
    avgdesc.compute_characs(pointfeats_arrayarray2, point_pos)
    avgdesc.compute_characs(pointfeats_listarray2, point_pos)
    avgdesc.compute_characs(pointfeats_listdict0, point_pos)
    avgdesc.compute_characs(pointfeats_listdict1, point_pos)

    avgdesc = AvgDescriptor('array')
    avgdesc.compute_characs(pointfeats_arrayarray0, point_pos)
    avgdesc.compute_characs(pointfeats_listarray0, point_pos)
    avgdesc.compute_characs(pointfeats_arrayarray1, point_pos)
    avgdesc.compute_characs(pointfeats_listarray1, point_pos)
    avgdesc.compute_characs(pointfeats_arrayarray2, point_pos)
    avgdesc.compute_characs(pointfeats_listarray2, point_pos)

    avgdesc = AvgDescriptor('listdict')
    avgdesc.compute_characs(pointfeats_listdict0, point_pos)
    avgdesc.compute_characs(pointfeats_listdict1, point_pos)

    avgdesc = AvgDescriptor('listarray')
    avgdesc.compute_characs(pointfeats_listarray0, point_pos)
    avgdesc.compute_characs(pointfeats_listarray1, point_pos)
    avgdesc.compute_characs(pointfeats_listarray2, point_pos)

    avgdesc = AvgDescriptor('arrayarray')
    avgdesc.compute_characs(pointfeats_arrayarray0, point_pos)
    avgdesc.compute_characs(pointfeats_arrayarray1, point_pos)
    avgdesc.compute_characs(pointfeats_arrayarray2, point_pos)

#    avgdesc.reducer(characs, point_pos)
#    avgdesc.reducer(characs, None)
#    avgdesc.reducer(feats, point_pos)
#    avgdesc.reducer(feats, None)
#    avgdesc.reducer(feats_dict, point_pos)
#    avgdesc.reducer(feats_dict, None)
#
#    avgdesc.aggdescriptor(characs, point_pos)
#    avgdesc.aggdescriptor(characs, None)
#    avgdesc.aggdescriptor(feats, point_pos)
#    avgdesc.aggdescriptor(feats, None)
#    avgdesc.aggdescriptor(feats_dict, point_pos)
#    avgdesc.aggdescriptor(feats_dict, None)

    # Not specific
    avgdesc.to_complete_measure(measure)
    #avgdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)
    avgdesc.set_global_info(None)
    avgdesc.set_functions(None, None)

    #################################
    #### Countdescriptor
#    point_pos = np.random.random((10, 5))
#    measure = np.random.random((100, 10, 2))
#    countdesc = Countdescriptor()
#    characs = np.random.randint(0, 10, 50).reshape((10, 5))
#    feats = categorical_array_features(100, 10)
#    feats_dict = categorical_dict_features(100, 10)

    countdesc = Countdescriptor()
    countdesc.compute_characs(pointfeats_arrayarray0, point_pos)
    countdesc.compute_characs(pointfeats_arrayarray1, point_pos)
    countdesc.compute_characs(pointfeats_listarray0, point_pos)
    countdesc.compute_characs(pointfeats_listarray1, point_pos)
#
#    countdesc.reducer(characs, point_pos)
#    countdesc.reducer(characs, None)
#    countdesc.reducer(feats, point_pos)
#    countdesc.reducer(feats, None)
#    countdesc.reducer(feats_dict, point_pos)
#    countdesc.reducer(feats_dict, None)
#
#    countdesc.aggdescriptor(characs, point_pos)
#    countdesc.aggdescriptor(characs, None)
#    countdesc.aggdescriptor(feats, point_pos)
#    countdesc.aggdescriptor(feats, None)
#    countdesc.aggdescriptor(feats_dict, point_pos)
#    countdesc.aggdescriptor(feats_dict, None)
#

    # Not specific
    countdesc.to_complete_measure(measure)
    #countdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)
    countdesc.set_global_info(None)
    countdesc._format_default_functions()
    countdesc.set_functions(None, None)
    countdesc.set_functions(None, 'dict')

#    #################################
#    #### Pjensen
#    pjensen = PjensenDescriptor()
#    # Specific
#    features = list(np.arange(20)) + list(np.random.randint(0, 20, 80))
#    features = np.array(features).reshape((100, 1))
#    pjensen.set_global_info(features)
#    feats = categorical_array_features(100, 20)
#    feats_dict = categorical_dict_features(100, 10)
#    characs = np.random.randint(0, 10, 50).reshape((10, 5))
#    point_pos = np.random.random((10, 5))
#    measure = np.random.randint(0, 50, 20*20).reshape((20, 20, 1))
#

#    pjensen = PjensenDescriptor(pointfeats_arrayarray0)
#    pjensen.compute_characs(pointfeats_arrayarray0, point_pos)
#    pjensen = PjensenDescriptor(pointfeats_arrayarray1)
#    pjensen.compute_characs(pointfeats_arrayarray1, point_pos)
#    pjensen = PjensenDescriptor(pointfeats_listarray0)
#    pjensen.compute_characs(pointfeats_listarray0, point_pos)
#    pjensen = PjensenDescriptor(pointfeats_listarray1)
#    pjensen.compute_characs(pointfeats_listarray1, point_pos)

#
#    # Functions
#    pjensen.compute_characs(characs, point_pos)
#    pjensen.compute_characs(characs, None)
#    pjensen.compute_characs(feats, point_pos)
#    pjensen.compute_characs(feats, None)
#    pjensen.compute_characs(feats_dict, point_pos)
#    pjensen.compute_characs(feats_dict, None)
#
#    pjensen.reducer(characs, point_pos)
#    pjensen.reducer(characs, None)
#    pjensen.reducer(feats, point_pos)
#    pjensen.reducer(feats, None)
#    pjensen.compute_characs(feats_dict, point_pos)
#    pjensen.compute_characs(feats_dict, None)
#
#    pjensen.aggdescriptor(characs, point_pos)
#    pjensen.aggdescriptor(characs, None)
#    pjensen.aggdescriptor(feats, point_pos)
#    pjensen.aggdescriptor(feats, None)
#    pjensen.compute_characs(feats_dict, point_pos)
#    pjensen.compute_characs(feats_dict, None)
#

#    # Not specific
#    pjensen.to_complete_measure(measure)
#    #pjensen.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)
#    pjensen._format_default_functions()
#    pjensen.set_functions(None, None)
#    pjensen.set_functions(None, 'dict')

#    #################################
#    #### SparseCounter
    # Only testing the specific functions. The others are tested in counter
    spcountdesc = SparseCounter()
#    spcountdesc.to_complete_measure(pointfeats_listdict0)
#    spcountdesc.to_complete_measure(pointfeats_listdict1)
##
##    spcountdesc.compute_characs(characs, point_pos)
##    spcountdesc.compute_characs(characs, None)
##
##    spcountdesc.reducer(characs, point_pos)
##    spcountdesc.reducer(characs, None)
##
##    spcountdesc.aggdescriptor(characs, point_pos)
##    spcountdesc.aggdescriptor(characs, None)
##
##    #spcountdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)
##
##    # Not specific
##    spcountdesc.set_global_info(None)
##    spcountdesc.set_functions(None, None)
#
#    #################################
#    #### NBinsHistogramDesc
    nbinsdesc = NBinsHistogramDesc(5)
#    characs = np.random.randint(0, 10, 50).reshape((10, 5))
#    point_pos = np.random.random((10, 5))
#    measure = np.random.random((100, 10, 2))
#    feats = categorical_array_features(100, 20)
#    feats_dict = categorical_dict_features(100, 10)
#
#    nbinsdesc.compute_characs(characs, point_pos)
#    nbinsdesc.compute_characs([characs], None)
#    nbinsdesc.compute_characs(feats, point_pos)
#    nbinsdesc.compute_characs(feats, None)
#    nbinsdesc.compute_characs(feats_dict, point_pos)
#    nbinsdesc.compute_characs(feats_dict, None)
#
#    nbinsdesc.reducer(characs, point_pos)
#    nbinsdesc.reducer(characs, None)
#    nbinsdesc.reducer(feats, point_pos)
#    nbinsdesc.reducer(feats, None)
#    nbinsdesc.reducer(feats_dict, point_pos)
#    nbinsdesc.reducer(feats_dict, None)
#
#    nbinsdesc.aggdescriptor(characs, point_pos)
#    nbinsdesc.aggdescriptor(characs, None)
#    nbinsdesc.aggdescriptor(feats, point_pos)
#    nbinsdesc.aggdescriptor(feats, None)
#    nbinsdesc.aggdescriptor(feats_dict, point_pos)
#    nbinsdesc.aggdescriptor(feats_dict, None)
#
#    #nbinsdesc.complete_desc_i(i, neighs_info, desc_i, desc_neighs, vals_i)
#
    # Specific
#    nbinsdesc.to_complete_measure(measure)
#    nbinsdesc._format_default_functions()
#    nbinsdesc.set_functions(None, None)
#    nbinsdesc.set_functions(None, 'dict')
    features = np.random.random((100, 5))
    nbinsdesc.set_global_info(features, True)
    nbinsdesc.set_global_info(features, False)
#    # Not specific

###############################################################################
###############################################################################
################################## TO TRASH ###################################
###############################################################################
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
