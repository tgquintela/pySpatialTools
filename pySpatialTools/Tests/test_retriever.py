

from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    CollectionRetrievers
import numpy as np
from scipy.sparse import coo_matrix


def random_sparse_matrix(density, shape, maxvalue=10):
    iss, jss, data = [], [], []
    for i in xrange(shape[0]):
        row = np.random.random(shape[1]) < density
        data.append(np.random.randint(0, maxvalue, row.sum()))
        jss.append(np.where(row)[0])
        iss.append(np.array([i]*row.sum()))
    data, iss, jss = np.hstack(data), np.hstack(iss), np.hstack(jss)
    matrix = coo_matrix((data, (iss, jss)), shape)
    return matrix


def test():
    n = 10000
    locs = np.random.random((n, 2))*100

    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, 3, ifdistance=True)

    gret = CollectionRetrievers([ret0, ret1])

    #reindices = np.arange(n).reshape((n, 1))
    #feat_arr0 = np.random.randint(0, 20, (n, 1))
    #feat_arr1 = np.random.random((n, 10))
    #avgdesc = AvgDescriptor(feat_arr1)
    #countdesc = Countdescriptor(feat_arr0)
    #import time
    #t0 = time.time()
    for i in xrange(n):
        neighs_info = ret0.retrieve_neighs(i)
        neighs_info = ret1.retrieve_neighs(i)
        neighs_info = gret.retrieve_neighs(i)
