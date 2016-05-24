
"""
test features_retriever
-----------------------
Testing the feature retriever.

"""

import numpy as np
from itertools import product
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager, _features_parsing_creation,\
    _featuresmanager_parsing_creation
from pySpatialTools.FeatureManagement.features_objects import\
    ImplicitFeatures, ExplicitFeatures, Features,\
    _featuresobject_parsing_creation
from pySpatialTools.FeatureManagement.descriptormodel import DummyDescriptor
from pySpatialTools.FeatureManagement.Descriptors import AvgDescriptor
from pySpatialTools.utils.perturbations import PermutationPerturbation
from pySpatialTools.utils.artificial_data import\
    categorical_agg_dict_features
from ..utils.util_classes import Neighs_Info


def test():
    ## Definition parameters
    n = 1000
    rei = 10

    n, n_feats = np.random.randint(10, 1000), np.random.randint(1, 20)
    ks = np.random.randint(1, 20)
    ###########################################################################
    ##########################
    #### FeatureRetriever testing
    reindices0 = np.arange(n)
    reindices = np.vstack([reindices0]+[np.random.permutation(n)
                                        for i in range(rei-1)]).T
    perturbation = PermutationPerturbation(reindices)

    aggcatfeats_dict = categorical_agg_dict_features(n, n_feats, ks)
    ## Impossible instantiation cases
    try:
        # Not valid oject as a feature
        boolean = False
        fm = FeaturesManager(None, None)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        # Object not valid as a feature
        boolean = False
        avgdesc = AvgDescriptor()
        fm = FeaturesManager([], avgdesc)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        Feat_imp = ImplicitFeatures(contfeats_ar0, perturbation)
#        fm = FeaturesManager(Feat_imp, None)
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
    try:
        # Different k_perturb
        boolean = False
        feats0 = ExplicitFeatures(np.random.random((100, 2, 4)))
        feats1 = ExplicitFeatures(np.random.random((100, 3, 3)))
        fm = FeaturesManager([feats0, feats1])
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        # Object not valid as a features
        boolean = False
        fm = FeaturesManager([5])
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        # Object not valid as a features
        boolean = False
        fm = FeaturesManager(lambda x: x)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    feats0 = np.random.random(100)
    feats1 = np.random.random((100, 1))
    feats2 = np.random.random((100, 1, 1))
    feats3 = np.random.random((100, 2))
    Feat_imp = ImplicitFeatures(feats1)
    Feat_imp2 = ImplicitFeatures(feats3, names=[3, 4])
    Feat_exp = ExplicitFeatures(aggcatfeats_dict)
    avgdesc = AvgDescriptor()

    pos_feats = [feats0, feats1, feats2, Feat_imp, Feat_exp,
                 [feats2, Feat_imp]]
    pos_mapvals_i = [None, ('matrix', 100, 20)]#, lambda x: x, 'matrix']
    pos_map_in = [None, lambda i_info, k: i_info]
    pos_map_out = [None, lambda self, feats: feats]
    pos_mode = [None, 'parallel', 'sequential']
    pos_desc = [None, avgdesc]

    possibilities = [pos_feats, pos_map_in, pos_map_out, pos_mapvals_i,
                     pos_mode, pos_desc]

    ## Random parameter space exploration
    mapper0 = [None]*3
    mapper1 = [(0, 0)]*3
    mapper2 = [np.array([np.zeros(100), np.zeros(100)]).T]*3
    mapper3 = [lambda idx: (0, 0)]*3
    mapper4 = [(0, 0), (0, 0), (1, 0)]
    pos_mappers = [mapper0, mapper1, mapper2, mapper3, mapper4]

    ## Information of indices
    nei_i = Neighs_Info()
#    nei_i.set(np.random.randint(0, 100, 5).reshape((5, 1, 1)))
    nei_i.set(np.random.randint(0, 100))
    nei_info = Neighs_Info()
    nei_info.set(np.random.randint(0, 100, 20).reshape((5, 2, 2)))

    ## Combinations
    for p in product(*possibilities):
#        ## Random exploration of parameters
#        feats = pos_feats[np.random.randint(0, len(pos_feats))]
#        m_input = pos_map_in[np.random.randint(0, len(pos_map_in))]
#        m_out = pos_map_out[np.random.randint(0, len(pos_map_out))]
#        m_vals_i = pos_mapvals_i[np.random.randint(0, len(pos_mapvals_i))]
#        mode = pos_mode[np.random.randint(0, len(pos_mode))]
#        desc = pos_desc[np.random.randint(0, len(pos_desc))]
        ## Exhaustive exploration of parameters
        i_selector = np.random.randint(0, len(pos_mappers))
        selectors = pos_mappers[i_selector]
        if i_selector == 4:
            continue
        #print i_selector
        feats, m_input, m_out, m_vals_i, mode, desc = p
        ## Instantiation
        fm = FeaturesManager(feats, maps_input=m_input, maps_output=m_out,
                             maps_vals_i=m_vals_i, mode=mode,
                             descriptormodels=desc, selectors=selectors)
        ## Basic parameters
        i0, i1 = 0, range(4)
        k_p = fm.k_perturb+1
        nei_info0 = Neighs_Info()
        nei_info1 = Neighs_Info()
        neis = np.random.randint(0, 100, 8*k_p).reshape((k_p, 4, 2))
        neis0 = np.random.randint(0, 100, 2*k_p).reshape((k_p, 1, 2))
        nei_info0.set(neis0)
        nei_info1.set(neis)
        ## Check basic functions
        fm[0]
        fm.shape
        len(fm)
        fm.nfeats
        fm.set_map_vals_i(m_vals_i)
        fm.initialization_desc()
        fm.initialization_output()
        fm.set_map_vals_i(100)
        fm.set_map_vals_i(m_vals_i)
        fm.set_descriptormodels(desc)
        ## Check basic functions
        fm.get_type_feats(0)
        fm.get_type_feats(50)
        fm.get_type_feats(0, tuple([(0, 0)]*3))
        fm.get_type_feats(50, tuple([(0, 0)]*3))
        # fm.get_type_feats(i0)
        # fm.get_type_feats(i1)

        t_feat_in, t_feat_out, t_feat_des = fm.get_type_feats(50)
        tf_in0, tf_out0, tf_desc0 = fm.get_type_feats(i0)
        tf_in1, tf_out1, tf_desc1 = fm.get_type_feats(i1)
        ## Interaction with featuresObjects
        # Input
        fm._get_input_features(50, k=range(k_p), typefeats=t_feat_in)
        if i_selector == 0:
            fm._get_input_features([50], k=range(k_p), typefeats=[(0, 0)])
        desc_i0 = fm._get_input_features(i0, k=range(k_p), typefeats=tf_in0)
        desc_i1 = fm._get_input_features(i1, k=range(k_p), typefeats=tf_in1)
#        print feats, len(desc_i0), k_p
#        print fm._get_input_features, desc_i0, desc_i1
        assert(len(desc_i0) == k_p)
        assert(len(desc_i1) == k_p)
        assert(len(desc_i0[0]) == 1)
        assert(len(desc_i1[0]) == 4)
        # Output
        fm._get_output_features(range(10), k=range(k_p), typefeats=t_feat_out)
        fm._get_output_features(neis[0], k=range(k_p), typefeats=t_feat_out)
        fm._get_output_features(neis, k=range(k_p), typefeats=t_feat_out)
        if i_selector == 0:
            fm._get_output_features([50], k=range(k_p), typefeats=[(0, 0)])
        desc_nei0 = fm._get_output_features(nei_info0, range(k_p), tf_out0)
        desc_nei1 = fm._get_output_features(nei_info1, range(k_p), tf_out1)
#        print fm._get_output_features
        assert(len(desc_nei0) == k_p)
        assert(len(desc_nei1) == k_p)
        assert(len(desc_nei0[0]) == 1)
        assert(len(desc_nei1[0]) == 4)
#        print desc_i0, desc_i1, desc_nei
#        print type(desc_i0), type(desc_i1), type(desc_nei)
#        print len(desc_i0), len(desc_i1), len(desc_nei)
        ## Interaction with map_vals_i
        fm._get_vals_i(20, range(k_p))
        fm._get_vals_i(range(20), range(k_p))
        vals_i0 = fm._get_vals_i(i0, range(k_p))
        vals_i1 = fm._get_vals_i(i1, range(k_p))
#        print fm._get_vals_i, vals_i0, vals_i1
        assert(len(vals_i0) == k_p)
        assert(len(vals_i1) == k_p)
        assert(len(vals_i0[0]) == 1)
        assert(len(vals_i1[0]) == 4)

        ## Completing features
        fm._complete_desc_i(i0, nei_info0, desc_i0, desc_nei0, vals_i0,
                            tf_desc0)
        fm._complete_desc_i(i1, nei_info1, desc_i1, desc_nei1, vals_i1,
                            tf_desc1)
        fm._complete_desc_i(i0, nei_info0, desc_i0, desc_nei0, vals_i0,
                            (1, 0))
        fm._complete_desc_i(i1, nei_info1, desc_i1, desc_nei1, vals_i1,
                            (1, 0))

        ## Computing altogether
        fm.compute_descriptors(i0, nei_info0)
        fm.compute_descriptors(i1, nei_info1)
        fm.compute_descriptors(i0, nei_info0, range(k_p))
        fm.compute_descriptors(i1, nei_info1, range(k_p))
#        fm.compute_descriptors(i0, range(10), range(k_p))
#        fm.compute_descriptors(i1, range(10), range(k_p))
        fm.compute_descriptors(i0, neis0[0], range(k_p))
        fm.compute_descriptors(i1, neis[0], range(k_p))
        fm.compute_descriptors(i0, neis0, range(k_p))
        fm.compute_descriptors(i1, neis, range(k_p))
        if i_selector == 0:
            fm.compute_descriptors([50], neis0[0], k=range(k_p),
                                   feat_selectors=[tuple([(0, 0)]*3)])
        # Strange cases
        if mode is None:
            FeaturesManager([ImplicitFeatures(feats1),
                             ImplicitFeatures(feats3, names=[3, 4])],
                            mode=mode)

    ## Cases
    feats = [ImplicitFeatures(feats1), ImplicitFeatures(feats1)]
    fm = FeaturesManager(feats, maps_input=m_input, maps_output=m_out,
                         maps_vals_i=m_vals_i, mode=mode,
                         descriptormodels=desc, selectors=selectors)
    if all([fea.typefeat == 'implicit' for fea in fm.features]):
        fm.add_perturbations(perturbation)

    ## Impossible function cases
    feats = [ImplicitFeatures(feats1), ImplicitFeatures(feats3, names=[3, 4])]
    try:
        ## Different variablesnames for sequential mode
        boolean = False
        fm = FeaturesManager(feats, maps_input=m_input, maps_output=m_out,
                             maps_vals_i=m_vals_i, mode='sequential',
                             descriptormodels=desc, selectors=selectors)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    try:
        boolean = False
        fm[-1]
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ###########################################################################
    #### Testing auxiliar parsing
    ## Functions which carry the uniformation of inputs from possible ways to
    ## input features information.
    ##
    feats0 = np.random.randint(0, 10, 100)
    feats1 = feats0.reshape((100, 1))
    feats2 = np.random.random((100, 2, 3))
    desc = DummyDescriptor()
    pars_feats = {}

    # Testing combinations of possible inputs
    feats_info = feats0
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    feats_info = feats1
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    feats_info = feats2
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    feats_info = (feats0, pars_feats)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    feats_info = (feats1, pars_feats)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    feats_info = (feats2, pars_feats)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    feats_info = (feats0, pars_feats, desc)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    pars_feats = {}
    feats_info = (feats1, pars_feats, desc)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    pars_feats = {}
    feats_info = (feats2, pars_feats, desc)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))

    features_obj = _featuresobject_parsing_creation(feats_info)
    assert(isinstance(features_obj, Features))
    features_ret = _features_parsing_creation(features_obj)
    assert(isinstance(features_ret, FeaturesManager))
    features_obj = _featuresobject_parsing_creation(feats_info)
    assert(isinstance(features_obj, Features))
    pars_feats = {}
    feats_info = (features_obj, pars_feats)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    pars_feats = {}
    feats_info = (features_obj, pars_feats, desc)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    pars_feats = {}
    feats_info = (features_obj, pars_feats, [desc, desc])
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))

    feats_info = ((feats0, {}), pars_feats)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    pars_feats = {}
    feats_info = ((feats0, {}), pars_feats, desc)
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    pars_feats = {}
    feats_info = ((feats0, {}), pars_feats, [desc, desc])
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))
    pars_feats = {}
    feats_info = ((feats0, {}, desc), pars_feats, [desc, desc])
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))

    feats_info = features_ret
    features_ret = _features_parsing_creation(feats_info)
    assert(isinstance(features_ret, FeaturesManager))

    features_ret = _featuresmanager_parsing_creation(features_obj)
    assert(isinstance(features_ret, FeaturesManager))
    features_ret = _featuresmanager_parsing_creation(features_ret)
    assert(isinstance(features_ret, FeaturesManager))

    feats_info = ((feats0, {}, desc), {})
    features_ret = _featuresmanager_parsing_creation(features_ret)
    assert(isinstance(features_ret, FeaturesManager))
    feats_info = ((feats0, {}, desc), {}, [desc, desc])
    features_ret = _featuresmanager_parsing_creation(features_ret)
    assert(isinstance(features_ret, FeaturesManager))
