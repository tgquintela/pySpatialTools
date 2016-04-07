
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
from ..utils.util_classes import Neighs_Info


def test():
    ## Definition parameters
    n = 1000
    m = 5
    rei = 10
    ## Definition arrays
    aggfeatures = np.random.random((n/2, m, rei))
    features0 = np.random.random((n, m))
    features1 = np.random.random((n, m))
    features2 = np.vstack([np.random.randint(0, 10, n) for i in range(m)]).T
    reindices0 = np.arange(n)
    reindices = np.vstack([reindices0]+[np.random.permutation(n)
                                        for i in range(rei-1)]).T
    perturbation = PermutationPerturbation(reindices)

    ## Definition classes
    Feat = ExplicitFeatures(aggfeatures)
    Feat0 = ImplicitFeatures(features0, perturbation)
    Feat1 = ImplicitFeatures(features1, perturbation)
    Feat2 = ImplicitFeatures(features2, perturbation)

    features_objects = [Feat0, Feat1, Feat2]
    avgdesc = AvgDescriptor()
    featret = FeaturesManager(features_objects, avgdesc)
    try:
        features_objects = [Feat, Feat0, Feat1, Feat2]
        featret = FeaturesManager(features_objects, avgdesc)
        raise Exception("It should not accept that inputs.")
    except:
        pass

    ## Other functions
    # Indexing
    Feat[0]
    Feat0[0]
    Feat1[0]
    Feat2[0]

    Feat[(0, 0)]
    Feat0[(0, 0)]
    Feat1[(0, 0)]
    Feat2[(0, 0)]
    Feat[([0], [0])]
    Feat0[([0], [0])]
    Feat1[([0], [0])]
    Feat2[([0], [0])]
    Feat[([0], [0.])]
    Feat0[([0], [0.])]
    Feat1[([0], [0.])]
    Feat2[([0], [0.])]

    Feat[0:3]
    Feat[:]
    Feat0[:]
    Feat1[:]
    Feat2[:]

    Feat[((0, 0), 0)]
    Feat0[((0, 0), 0)]
    Feat1[((0, 0), 0)]
    Feat2[((0, 0), 0)]

    Feat[(([0], [0]), [0])]
    Feat0[(([0], [0]), [0])]
    Feat1[(([0], [0]), [0])]
    Feat2[(([0], [0]), [0])]

    Feat0[[0, 4, 5]]
    Feat1[[0, 4, 5]]
    Feat2[[0, 4, 5]]

    nei = Neighs_Info()
    nei.set((([0], [0]), [0]))
    Feat[nei]
    Feat0[nei]
    Feat1[nei]
    Feat2[nei]

    # shape
#    Feat.shape
    Feat0.shape
    Feat1.shape
    Feat2.shape

    featret[0]
    featret.shape
    featret.nfeats
    len(featret)

    ## Empty call
    Feat0[(([], []), [0])]
    Feat1[(([], []), [0])]
    Feat2[(([], []), [0])]

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
