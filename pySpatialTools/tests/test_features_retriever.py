
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


def test():
    ## Definition parameters
    n = 1000
    m = 5
    rei = 10
    ## Definition arrays
    aggfeatures = np.random.random((n/2, m, rei))
    features0 = np.random.random((n/5, m))
    features1 = np.random.random((n/3, m))
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
