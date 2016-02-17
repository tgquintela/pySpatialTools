
"""
artificial data
----------------
functions which creates artificial data.
"""

from scipy.sparse import coo_matrix
import numpy as np
import networkx as nx

from pySpatialTools.SpatialRelations import RegionDistances
from pySpatialTools.Discretization import SetDiscretization


def random_membership(n_elements, n_collections, multiple=True):
    """Function to create membership relations for set discretizor."""
    if multiple:
        membership = []
        for i in xrange(n_elements):
            aux = np.random.randint(0, n_collections,
                                    np.random.randint(2*n_collections))
            aux = np.unique(aux)
            membership.append(aux)
    else:
        membership = np.random.randint(0, n_collections, n_elements)
    set_disc = SetDiscretization(membership)
    return set_disc


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


def generate_random_relations(n, sym=True, store='network'):
    indices = np.where(np.random.random(n*n) > 0.5)[0]
    data = np.random.random(indices.shape[0])
    indices = (indices / n, indices % n)
    sparse = coo_matrix((data, indices), shape=(n, n))
    if sym:
        sparse = (sparse + sparse.T)/2
    regs, i = [], 0
    while True:
        if np.random.random() > 0.9:
            regs.append(i)
        if len(regs) == n:
            break
        i += 1
    regs = np.array(regs).reshape((n, 1))

    sp_relations = RegionDistances()
    sp_relations.data = regs
    if store == 'network':
        net = nx.from_scipy_sparse_matrix(sparse)
        mapping = dict(zip(net.nodes(), regs.ravel()))
        sp_relations.relations = nx.relabel_nodes(net, mapping)
    elif store == 'sparse':
        sp_relations.relations = sparse
    return sp_relations