
"""

"""

from pySpatialTools.Feature_engineering.Descriptors import Countdescriptor,\
    AvgDescriptor
from pySpatialTools.Feature_engineering import SpatialDescriptorModel
from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    CollectionRetrievers
import numpy as np


def test():
    n = 10000
    locs = np.random.random((n, 2))*100

    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, 3, ifdistance=True)
    gret = CollectionRetrievers([ret0, ret1])

    avgdesc = AvgDescriptor(feat_arr1)
    countdesc = Countdescriptor(feat_arr0)

    gdesc = 
    sp_model0 = SpatialDescriptorModel(gret, gdesc)
