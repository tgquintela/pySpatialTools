
"""
Testing sampling


"""

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from pySpatialTools.Retrieve.Spatial_Relations import CenterLocsRegionDistances


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

    sp_relations = CenterLocsRegionDistances()
    sp_relations.data = regs
    if store == 'network':
        net = nx.from_scipy_sparse_matrix(sparse)
        mapping = dict(zip(net.nodes(), regs.ravel()))
        sp_relations.relations = nx.relabel_nodes(net, mapping)
    elif store == 'sparse':
        sp_relations.relations = sparse
    return sp_relations


def test():
    ## Generate sp_relations
    sp_relations = generate_random_relations(100)
    connected = nx.connected_components(sp_relations.relations)

