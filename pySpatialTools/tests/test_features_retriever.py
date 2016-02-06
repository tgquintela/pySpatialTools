
"""
"""

from pySpatialTools.Feature_engineering.features_retriever import Features,\
    AggFeatures, FeaturesRetriever, PointFeatures
import numpy as np
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    PointFeaturePertubation


def test():
    ## Definition parameters
    n = 1000
    m = 5
    rei = 10
    ## Definition arrays
    aggfeatures = np.random.random((n/2, m, rei))
    features0 = np.random.random((n/5, m))
    features1 = np.random.random((n/3, m/3))
    features2 = np.vstack([np.random.randint(0, 10, n) for i in range(m)]).T
    reindices0 = np.arange(n)
    reindices = np.vstack([reindices0]+[np.random.permutation(n)
                                        for i in range(rei-1)]).T
    perturbation = PermutationPerturbation(reindices)

    ## Definition classes
    Feat = AggFeatures(aggfeatures)
    Feat0 = PointFeatures(features0, perturbation)
    Feat1 = PointFeatures(features1, perturbation)
    Feat2 = PointFeatures(features2, perturbation)

    features_objects = [Feat0, Feat1, Feat2]
    featret = FeaturesRetriever(features_objects)
    try:
        features_objects = [Feat, Feat0, Feat1, Feat2]
        featret = FeaturesRetriever(features_objects)
        raise Exception("It should not accept that inputs.")
    except:
        pass
