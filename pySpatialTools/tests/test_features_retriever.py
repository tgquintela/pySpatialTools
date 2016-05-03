
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

    class DummyDesc:
        def set_functions(self, typefeatures, outformat):
            pass

    class Dummy1Desc(DummyDesc):
        def __init__(self):
            self.compute_characs = None
            self._out_formatter = None
            self.reducer = None

    class Dummy2Desc_exp(DummyDesc):
        def __init__(self):
            self.compute_characs = lambda x, d: [e[0] for e in x]
            self._out_formatter = lambda x, y1, y2, y3: x
            self.reducer = None

    class Dummy2Desc_imp(DummyDesc):
        def __init__(self):
            self.compute_characs = lambda x, d: np.array([e[0] for e in x])
            self._out_formatter = lambda x, y1, y2, y3: x
            self.reducer = None

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
            print 'x'*100, Feat.k_perturb, Feat.shape
            Feat[(([[0], [0]], [[0], [0]]), [0, 1])]
        Feat[[0, 4, 5]]
        try:
            boolean = False
            Feat[-1]
            boolean = True
        except:
            if boolean:
                raise Exception("It has to halt here.")
        try:
            boolean = False
            feats = Feat._retrieve_feats([[[0]]], -1, None)
            boolean = True
        except:
            if boolean:
                raise Exception("It has to halt here.")
        try:
            boolean = False
            Feat._retrieve_feats([[[0]]], 10000, None)
            boolean = True
        except:
            if boolean:
                raise Exception("It has to halt here.")
        try:
            boolean = False
            Feat[len(Feat)]
            boolean = True
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

#    Feat = ExplicitFeatures(aggcontfeats_ar0)
#    test_getitem(Feat)
#    Feat = ExplicitFeatures(aggcatfeats_ar0)
#    test_getitem(Feat)
#    Feat = ExplicitFeatures(aggcatfeats_ar1)
#    test_getitem(Feat)
#    Feat = ExplicitFeatures(aggcontfeats_dict)
#    test_getitem(Feat)
#    Feat = ExplicitFeatures(aggcatfeats_dict)
#    test_getitem(Feat)

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
        if p[0] < 3:
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

#    Feat0 = ImplicitFeatures(features0, perturbation)
#    Feat1 = ImplicitFeatures(features1, perturbation)
#    Feat2 = ImplicitFeatures(features2, perturbation)
#
#    features_objects = [Feat0, Feat1, Feat2]
#    featret = FeaturesManager(features_objects, avgdesc)
#    try:
#        boolean = False
#        features_objects = [Feat, Feat0, Feat1, Feat2]
#        featret = FeaturesManager(features_objects, avgdesc)
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It should not accept that inputs.")

#    contfeats_ar0 = continuous_array_features(n, n_feats)
#    catfeats_ar0 = categorical_array_features(n, n_feats)
#    catfeats_ar1 = categorical_array_features(n, n_feats2)
#    contfeats_dict = continuous_dict_features(n, n_feats)
#    catfeats_dict = categorical_dict_features(n, n_feats)
#
#    Feat = ImplicitFeatures(contfeats_ar0)
#    test_getitem(Feat)
#    Feat = ImplicitFeatures(catfeats_ar0)
#    test_getitem(Feat)
#    Feat = ImplicitFeatures(catfeats_ar1)
#    test_getitem(Feat)
#    Feat = ImplicitFeatures(contfeats_dict)
#    test_getitem(Feat)
#    Feat = ImplicitFeatures(catfeats_dict)
#    test_getitem(Feat)


    ## Other functions
    # Indexing
#    Feat[0]
#    Feat0[0]
#    Feat1[0]
#    Feat2[0]
#
#    Feat[(0, 0)]
#    Feat0[(0, 0)]
#    Feat1[(0, 0)]
#    Feat2[(0, 0)]
#    Feat[([0], [0])]
#    Feat0[([0], [0])]
#    Feat1[([0], [0])]
#    Feat2[([0], [0])]
#    Feat[([0], [0.])]
#    Feat0[([0], [0.])]
#    Feat1[([0], [0.])]
#    Feat2[([0], [0.])]
#
#    Feat[0:3]
#    Feat[:]
#    Feat0[:]
#    Feat1[:]
#    Feat2[:]
#
#    Feat[((0, 0), 0)]
#    Feat0[((0, 0), 0)]
#    Feat1[((0, 0), 0)]
#    Feat2[((0, 0), 0)]
#
#    Feat[(([0], [0]), [0])]
#    Feat0[(([0], [0]), [0])]
#    Feat1[(([0], [0]), [0])]
#    Feat2[(([0], [0]), [0])]
#
#    Feat0[[0, 4, 5]]
#    Feat1[[0, 4, 5]]
#    Feat2[[0, 4, 5]]
#
#    try:
#        boolean = False
#        Feat[-1]
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        Feat0[len(Feat0)]
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        Feat1[len(Feat1)]
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        Feat2[len(Feat2)]
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#
#    nei = Neighs_Info()
#    nei.set((([0], [0]), [0]))
#    Feat[nei]
#    Feat0[nei]
#    Feat1[nei]
#    Feat2[nei]
#
#    nei = Neighs_Info()
#    nei.set([[[0, 4], [0, 3]]])
#    Feat[nei]
#    Feat0[nei]
#    Feat1[nei]
#    Feat2[nei]
#
#    nei = Neighs_Info(staticneighs=True)
#    nei.set([[0, 4], [0, 3]])
#    Feat1[nei, 1]
#
#    # shape
##    Feat.shape
#    Feat0.shape
#    Feat1.shape
#    Feat2.shape
#
#    featret[0]
#    featret.shape
#    featret.nfeats
#    len(featret)
#
#    ## Empty call
#    Feat0[(([[]], [[]]), [0])]
#    Feat1[(([[]], [[]]), [0])]
#    Feat2[(([[]], [[]]), [0])]
#
#    Feat.set_descriptormodel(avgdesc)
#    Feat0.set_descriptormodel(avgdesc)
#    Feat1.set_descriptormodel(avgdesc)
#    Feat2.set_descriptormodel(avgdesc)

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

#    Feat.set_descriptormodel(avgdesc)

#    nei = Neighs_Info()
#    nei.set((([0], [0]), [0]))
#    Feat[nei]
#    nei = Neighs_Info()
#    nei.set([[[0, 4], [0, 3]]])
#    Feat[nei]
