
"""
artificial spatial relations
----------------------------
Module which groups all the functions related with artificial spatial
relations.

"""

from scipy.sparse import coo_matrix
import numpy as np
import networkx as nx

from pySpatialTools.SpatialRelations import RegionDistances


def randint_sparse_matrix(density, shape, maxvalue=10):
    iss, jss, data = [], [], []
    for i in xrange(shape[0]):
        row = np.random.random(shape[1]) < density
        data.append(np.random.randint(0, maxvalue, row.sum()))
        jss.append(np.where(row)[0])
        iss.append(np.array([i]*row.sum()))
    data, iss, jss = np.hstack(data), np.hstack(iss), np.hstack(jss)
    matrix = coo_matrix((data, (iss, jss)), shape)
    return matrix


def generate_randint_relations(density, shape, p0=0., maxvalue=1):
    sparse = randint_sparse_matrix(density, shape, maxvalue)
    data_in, data_out, i = [], [], 0
    while True:
        if len(data_in) != shape[0]:
            if np.random.random() > p0:
                data_in.append(i)
        if len(data_out) != shape[1]:
            if np.random.random() > p0:
                data_out.append(i)
        if len(data_in) != shape[0] and len(data_out) != shape[1]:
            break
        i += 1
    sp_relations = RegionDistances(sparse, _data=data_out, data_in=data_in)
    return sp_relations


def generate_random_relations_cutoffs(n, p0=0.5, p1=0.9, sym=True,
                                      store='network'):
    indices = np.where(np.random.random(n*n) > p0)[0]
    data = np.random.random(indices.shape[0])
    indices = (indices / n, indices % n)
    sparse = coo_matrix((data, indices), shape=(n, n))
    if sym:
        sparse = (sparse + sparse.T)/2
    regs, i = [], 0
    while True:
        if np.random.random() < p1:
            regs.append(i)
        if len(regs) == n:
            break
        i += 1
    regs = np.array(regs).reshape((n, 1))
    if store == 'network':
        net = nx.from_scipy_sparse_matrix(sparse)
        mapping = dict(zip(net.nodes(), regs.ravel()))
        sp_relations = RegionDistances(nx.relabel_nodes(net, mapping),
                                       _data=regs)
    elif store == 'sparse':
        sp_relations = RegionDistances(sparse, _data=regs)
    return sp_relations
