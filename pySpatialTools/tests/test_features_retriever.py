
"""
test features_retriever
-----------------------
Testing the feature retriever.

"""

from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import Features,\
    ImplicitFeatures, ExplicitFeatures
from pySpatialTools.FeatureManagement.Descriptors import AvgDescriptor
import numpy as np
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

    n, n_feats = np.random.randint(1, 1000), np.random.randint(1, 20)
    n_feats2 = [np.random.randint(1, 20) for i in range(n_feats)]
    ks = np.random.randint(1, 20)

    avgdesc = AvgDescriptor()

    ### Test functions definitions
    def test_getfeatsk(Feat):
        nei = Neighs_Info()
        nei.set((([0], [0]), [0]))
        i, d, _, k = 0
        pass

    def test_getitem(Feat):
        k = 0
        idxs = np.random.randint(0, 5, 20).reshape((1, 4, 5))
        #Feat._get_feats_k(idxs, k)
        #Feat._get_feats_k(list(idxs), k)
        Feat[0]
        Feat[(0, 0)]
        Feat[([0], [0])]
        Feat[([0], [0.])]
        Feat[0:3]
        Feat[:]
        Feat[((0, 0), 0)]
        Feat[(([0], [0]), [0])]
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
        #Feat.set_descriptormodel(avgdesc)

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

    aggcontfeats_ar0 = continuous_agg_array_features(n, n_feats, ks)
    aggcatfeats_ar0 = categorical_agg_array_features(n, n_feats, ks)
    aggcatfeats_ar1 = categorical_agg_array_features(n, n_feats2, ks)
    aggcontfeats_dict = continuous_agg_dict_features(n, n_feats, ks)
    aggcatfeats_dict = categorical_agg_dict_features(n, n_feats, ks)

    Feat = ExplicitFeatures(aggcontfeats_ar0)
    test_getitem(Feat)
    Feat = ExplicitFeatures(aggcatfeats_ar0)
    test_getitem(Feat)
    Feat = ExplicitFeatures(aggcatfeats_ar1)
    test_getitem(Feat)
    Feat = ExplicitFeatures(aggcontfeats_dict)
    test_getitem(Feat)
    Feat = ExplicitFeatures(aggcatfeats_dict)
    test_getitem(Feat)

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

    Feat_imp = ImplicitFeatures(contfeats_ar0, perturbation)
    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(catfeats_ar0, perturbation)
    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(catfeats_ar1, perturbation)
    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(contfeats_dict, perturbation)
#    test_getitem(Feat_imp)
    Feat_imp = ImplicitFeatures(catfeats_dict, perturbation)
#    test_getitem(Feat_imp)

    Feat0 = ImplicitFeatures(features0, perturbation)
    Feat1 = ImplicitFeatures(features1, perturbation)
    Feat2 = ImplicitFeatures(features2, perturbation)
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
#
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
