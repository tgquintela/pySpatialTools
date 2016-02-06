

from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    CollectionRetrievers
import numpy as np
from scipy.sparse import coo_matrix

from pySpatialTools.utils.artificial_data import random_sparse_matrix


def test():
    n = 1000
    locs = np.random.random((n, 2))*100

    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, 3, ifdistance=True)

    gret = CollectionRetrievers([ret0, ret1])

    for i in xrange(n):
        neighs_info = ret0.retrieve_neighs(i)
        neighs_info = ret1.retrieve_neighs(i)
        neighs_info = gret.retrieve_neighs(i)
