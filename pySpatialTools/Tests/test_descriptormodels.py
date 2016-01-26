

from pySpatialTools.Feature_engineering.Descriptors import\
    Countdescriptor, AvgDescriptor
from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    CollectionRetrievers
from pySpatialTools.Feature_engineering import SpatialDescriptorModel

import numpy as np


n = 10000
locs = np.random.random((n, 2))*100
feat_arr0 = np.random.randint(0, 20, (n, 1))
feat_arr1 = np.random.random((n, 10))
reindices = np.arange(n).reshape((n, 1))
ret0 = KRetriever(locs, 3, ifdistance=True)
ret1 = CircRetriever(locs, 3, ifdistance=True)
gret = CollectionRetrievers([ret0, ret1])

countdesc = Countdescriptor(feat_arr0, ('matrix', n))
sp_model0 = SpatialDescriptorModel(gret, countdesc)


def test():
    n = 10000
    locs = np.random.random((n, 2))*100
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))
    reindices = np.arange(n).reshape((n, 1))

    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, 3, ifdistance=True)


    countdesc = Countdescriptor(feat_arr0)

    avgdesc = AvgDescriptor(feat_arr1)

    sp_model0 = SpatialDescriptorModel(ret0, avgdesc)
    sp_model0.set(reindices=reindices)
    sp_model1 = SpatialDescriptorModel(ret1, avgdesc)
    sp_model1.set(reindices=reindices)
    sp_model2 = SpatialDescriptorModel(ret0, countdesc)
    sp_model2.set(reindices=reindices)
    sp_model3 = SpatialDescriptorModel(ret1, countdesc)
    sp_model3.set(reindices=reindices)
    sp_model4 = SpatialDescriptorModel([ret0, ret1], countdesc)
    sp_model4.set(reindices=reindices, cond_agg=np.random.randint(0, 2, n))

    corr = sp_model0.compute_net()
    corr = sp_model1.compute_net()
    corr = sp_model2.compute_net()
    corr = sp_model3.compute_net()
    corr = sp_model4.compute_net()
