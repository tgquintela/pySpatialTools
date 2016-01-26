
"""
"""

from pySpatialTools.Feature_engineering.features_retriever import Features,\
    AggFeatures, FeaturesRetriever, PointFeatures
import numpy as np


def test():
    ## Definition parameters
    n = 100000
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
    ## Definition classes
    Feat = AggFeatures(aggfeatures)
    Feat0 = PointFeatures(features0, reindices)
    Feat1 = PointFeatures(features1, reindices)
    Feat2 = PointFeatures(features2, reindices)

    features_objects = [Feat, Feat0, Feat1, Feat2]
    featret = FeaturesRetriever(features_objects)
