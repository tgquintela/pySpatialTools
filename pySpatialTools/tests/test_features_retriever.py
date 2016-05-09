
"""
test features_retriever
-----------------------
Testing the feature retriever.

"""

import numpy as np
from itertools import product
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import Features,\
    ImplicitFeatures, ExplicitFeatures
from pySpatialTools.FeatureManagement.Descriptors import AvgDescriptor
from pySpatialTools.utils.perturbations import PermutationPerturbation
from pySpatialTools.utils.artificial_data import continuous_array_features,\
    categorical_array_features, continuous_dict_features,\
    categorical_dict_features, continuous_agg_array_features,\
    categorical_agg_array_features, continuous_agg_dict_features,\
    categorical_agg_dict_features
from ..utils.util_classes import Neighs_Info


def test():
    ## Definition parameters
    n = 1000
    m = 5
    rei = 10

    n, n_feats = np.random.randint(10, 1000), np.random.randint(1, 20)
    n_feats2 = [np.random.randint(1, 20) for i in range(n_feats)]
    ks = np.random.randint(1, 20)

    def create_ids(n1):
        aux = np.random.randint(1, 4, n1)
        return np.cumsum(aux)

    def create_featurenames(n1):
        aux = create_ids(n1)
        return [str(e) for e in aux]

    def extract_featurenames_agg(aggdictfeats):
        names = []
        for k in range(len(aggdictfeats)):
            names += extract_featurenames(aggdictfeats[k])
        names = list(set(names))
        return names

    def extract_featurenames(aggdictfeats):
        names = []
        for i in range(len(aggdictfeats)):
            names += aggdictfeats[i].keys()
        names = list(set(names))
        return names

    def compute_featurenames(features):
        names = []
        if type(features) == np.ndarray:
            names = [str(e) for e in range(len(features[0]))]
        return names

    class DummyDesc:
        def set_functions(self, typefeatures, outformat):
            pass

    class Dummy1Desc(DummyDesc):
        def __init__(self):
            self.compute_characs = None
            self._out_formatter = None
            self.reducer = None
            self._f_default_names = compute_featurenames

    class Dummy2Desc_exp(DummyDesc):
        def __init__(self):
            self.compute_characs = lambda x, d: [e[0] for e in x]
            self._out_formatter = lambda x, y1, y2, y3: x
            self.reducer = None
            self._f_default_names = compute_featurenames

    class Dummy2Desc_imp(DummyDesc):
        def __init__(self):
            self.compute_characs = lambda x, d: np.array([e[0] for e in x])
            self._out_formatter = lambda x, y1, y2, y3: x
            self.reducer = None
            self._f_default_names = compute_featurenames

    ## Possible descriptormodels to test
    avgdesc = AvgDescriptor()
    dum1desc = Dummy1Desc()
    dum2desc = Dummy2Desc_imp()
    dum2desc_agg = Dummy2Desc_exp()

    ### Test functions definitions
    def test_getfeatsk(Feat):
        nei = Neighs_Info()
        nei.set((([0], [0]), [0]))
        i, d, _, k = 0
        pass

    def test_getitem(Feat):
        #k = 0
        #idxs = np.random.randint(0, 5, 20).reshape((1, 4, 5))
        #Feat._get_feats_k(idxs, k)
        #Feat._get_feats_k(list(idxs), k)
        #Feat[[]]
        Feat[0]
        Feat[(0, 0)]
        Feat[([0], [0])]
        Feat[([0], [0.])]
        Feat[0:3]
        Feat[:]
        Feat[((0, 0), 0)]
        Feat[(([0], [0]), [0])]
        if Feat.k_perturb:
#            print 'x'*100, Feat.k_perturb, Feat.shape
            Feat[(([[0], [0]], [[0], [0]]), [0, 1])]
        Feat[[0, 4, 5]]
        try:
            boolean = False
            Feat[-1]
            boolean = True
            raise Exception("It has to halt here.")
        except:
            if boolean:
                raise Exception("It has to halt here.")
        try:
            boolean = False
            feats = Feat._retrieve_feats([[[0]]], -1, None)
            boolean = True
            raise Exception("It has to halt here.")
        except:
            if boolean:
                raise Exception("It has to halt here.")
        try:
            boolean = False
            Feat._retrieve_feats([[[0]]], 10000, None)
            boolean = True
            raise Exception("It has to halt here.")
        except:
            if boolean:
                raise Exception("It has to halt here.")
        try:
            boolean = False
            Feat[len(Feat)]
            boolean = True
            raise Exception("It has to halt here.")
        except:
            if boolean:
                raise Exception("It has to halt here.")
        nei = Neighs_Info()
        nei.set((([0], [0]), [0]))
        Feat[nei]
        nei = Neighs_Info()
        nei.set([[[0, 4], [0, 3]]])
        Feat[nei]
        nei = Neighs_Info(staticneighs=True)
        nei.set([[0, 4], [0, 3]])
        Feat[nei, 0]
        # shape
        Feat.shape
        ## Empty call
        Feat[(([[]], [[]]), [0])]
        # Descriptormodels setting
        # null formatters
        #Feat._format_characterizer(None, None)
        Feat.set_descriptormodel(dum1desc)
        if Feat.typefeat == 'implicit':
            Feat.set_descriptormodel(dum2desc)
        else:
            Feat.set_descriptormodel(dum2desc_agg)

        avgdesc = AvgDescriptor()
        Feat.set_descriptormodel(avgdesc)

    ## Definition arrays
    aggfeatures = np.random.random((n/2, m, rei))
    features0 = np.random.random((n, m))
    features1 = np.random.random((n, m))
    features2 = np.vstack([np.random.randint(0, 10, n) for i in range(m)]).T
    reindices0 = np.arange(n)
    reindices = np.vstack([reindices0]+[np.random.permutation(n)
                                        for i in range(rei-1)]).T
    perturbation = PermutationPerturbation(reindices)

    ###########################################################################
    ##########################
    #### Explicit Features testing
    ### Definition classes
    # Instantiation
    Feat = ExplicitFeatures(np.random.randint(0, 20, 100))
    test_getitem(Feat)
    Feat = ExplicitFeatures(np.random.random((100, 2)))
    test_getitem(Feat)
    Feat = ExplicitFeatures(aggfeatures)
    test_getitem(Feat)
    try:
        boolean = False
        ExplicitFeatures(np.random.random((10, 1, 1, 1)))
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It should not accept that inputs.")

    ## Exhaustive instantiation testing
    aggcontfeats_ar0 = continuous_agg_array_features(n, n_feats, ks)
    aggcatfeats_ar0 = categorical_agg_array_features(n, n_feats, ks)
    aggcatfeats_ar1 = categorical_agg_array_features(n, n_feats2, ks)
    aggcontfeats_dict = continuous_agg_dict_features(n, n_feats, ks)
    aggcatfeats_dict = categorical_agg_dict_features(n, n_feats, ks)

    pos_feats = [aggcontfeats_ar0, aggcatfeats_ar0, aggcatfeats_ar1,
                 aggcontfeats_dict, aggcatfeats_dict]
    pos_names = [create_featurenames(n_feats), create_featurenames(1),
                 create_featurenames(len(n_feats2)),
                 create_featurenames(n_feats),
                 extract_featurenames_agg(aggcontfeats_dict),
                 extract_featurenames_agg(aggcatfeats_dict)]
    pos_nss = [0, 1, 2, 3, 4]
    pos_null = [None, 0., np.inf]
    pos_characterizer = [None]
    pos_outformatter = [None]
    pos_indices = [None]

    possibilities = [pos_nss, pos_null, pos_characterizer, pos_outformatter,
                     pos_indices]

    for p in product(*possibilities):
#        print p
        ## Names definition
        names = []
        if np.random.randint(0, 2):
            names = pos_names[p[0]]
        ## Instantiation
        Feat = ExplicitFeatures(pos_feats[p[0]], names=names, indices=p[4],
                                characterizer=p[2], out_formatter=p[3],
                                nullvalue=p[1])
        ## Testing main functions
        test_getitem(Feat)

    ## Particular cases
    try:
        boolean = False
        names = [str(i) for i in range(len(aggcontfeats_ar0[0])+1)]
        ExplicitFeatures(aggcontfeats_ar0, names=names, indices=p[4],
                         characterizer=p[2], out_formatter=p[3],
                         nullvalue=p[1])
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    ##
    ## List features
    listfeatures = []
    for k in range(5):
        listfeatures_k = []
        for i in range(100):
            aux = np.unique(np.random.randint(0, 100, np.random.randint(5)))
            d = dict(zip(aux, np.random.random(len(aux))))
            listfeatures_k.append(d)
        listfeatures.append(listfeatures_k)
    Feat = ExplicitFeatures(listfeatures)
    len(Feat)
    nei = Neighs_Info()
    nei.set((([0], [0]), [0]))
    Feat[nei]
    nei = Neighs_Info()
    nei.set([[[0, 4], [0, 3]]])
    Feat[nei]

    ###########################################################################
    ##########################
    #### Implicit Features testing
    ### Definition classes
    # Instantiation
    contfeats_ar0 = continuous_array_features(n, n_feats)
    catfeats_ar0 = categorical_array_features(n, n_feats)
    catfeats_ar1 = categorical_array_features(n, n_feats2)
    contfeats_dict = continuous_dict_features(n, n_feats)
    catfeats_dict = categorical_dict_features(n, n_feats)

    pos_feats = [contfeats_ar0, catfeats_ar0, catfeats_ar1,
                 contfeats_dict, catfeats_dict]
    pos_names = [create_featurenames(n_feats), create_featurenames(1),
                 create_featurenames(len(n_feats2)),
                 create_featurenames(n_feats),
                 extract_featurenames(contfeats_dict),
                 extract_featurenames(catfeats_dict)]
    pos_nss = [0, 1, 2, 3, 4]
    pos_null = [None]  # TODO: [None, 0., np.inf]
    pos_characterizer = [None]
    pos_outformatter = [None]
    pos_indices = [None]  # TODO
    pos_perturbations = [None, perturbation]

    possibilities = [pos_nss, pos_null, pos_characterizer, pos_outformatter,
                     pos_indices, pos_perturbations]
    ## Combination of inputs testing
    for p in product(*possibilities):
#        print p
        ## Names definition
        names = []
        if np.random.randint(0, 2):
            names = pos_names[p[0]]
        ## Instantiation
        Feat = ImplicitFeatures(pos_feats[p[0]], names=names,
                                characterizer=p[2], out_formatter=p[3],
                                perturbations=p[5])
        ## Testing main functions
#        if p[0] < 3:
#            test_getitem(Feat)
        test_getitem(Feat)

    Feat_imp = ImplicitFeatures(contfeats_ar0, perturbation)
    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(catfeats_ar0, perturbation)
    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(catfeats_ar0.ravel(), perturbation, names=[0])
    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(catfeats_ar1, perturbation)
    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(contfeats_dict, perturbation)
#    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(catfeats_dict, perturbation)
#    test_getitem(Feat_imp)

    try:
        boolean = False
        Feat = ImplicitFeatures(contfeats_ar0, None)
        Feat._map_perturb(-1)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        Feat = ImplicitFeatures(contfeats_ar0, perturbation)
        Feat._map_perturb(-1)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        Feat = ImplicitFeatures(contfeats_ar0, perturbation)
        Feat._map_perturb(1000)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ###########################################################################
    ##########################
    #### FeatureRetriever testing
    reindices0 = np.arange(n)
    reindices = np.vstack([reindices0]+[np.random.permutation(n)
                                        for i in range(rei-1)]).T
    perturbation = PermutationPerturbation(reindices)

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
    pos_mappers = [mapper0, mapper1, mapper2, mapper3]

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
        selectors = pos_mappers[np.random.randint(0, len(pos_mappers))]
        feats, m_input, m_out, m_vals_i, mode, desc = p
        ## Instantiation
        fm = FeaturesManager(feats, maps_input=m_input, maps_output=m_out,
                             maps_vals_i=m_vals_i, mode=mode,
                             descriptormodels=desc, selectors=selectors)
        # Check basic functions
        fm[0]
        fm.shape
        len(fm)
        fm.nfeats
        fm.set_map_vals_i(m_vals_i)
        fm.initialization_desc()
        fm.initialization_output()
        fm.set_map_vals_i(100)
        fm.set_map_vals_i(m_vals_i)
        # Strange cases
        if mode is None:
            FeaturesManager([ImplicitFeatures(feats1),
                             ImplicitFeatures(feats3, names=[3, 4])],
                            mode=mode)
        fm.get_type_feat(0, [(0, 0)]*3)
        fm.get_type_feat(50, [(0, 0)]*3)
        fm.get_type_feat(50)
        fm.set_descriptormodels(desc)

    feats = [ImplicitFeatures(feats1), ImplicitFeatures(feats1)]
    fm = FeaturesManager(feats, maps_input=m_input, maps_output=m_out,
                         maps_vals_i=m_vals_i, mode=mode,
                         descriptormodels=desc, selectors=selectors)
    if all([fea.typefeat == 'implicit' for fea in fm.features]):
        fm.add_perturbations(perturbation)

    ## Impossible function cases
    feats = [ImplicitFeatures(feats1), ImplicitFeatures(feats3, names=[3, 4])]
    try:
        ## Different variablesnames
        boolean = False
        fm = FeaturesManager(feats, maps_input=m_input, maps_output=m_out,
                             maps_vals_i=m_vals_i, mode=mode,
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
