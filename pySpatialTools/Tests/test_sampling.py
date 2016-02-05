
"""
Testing sampling


"""

import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from pySpatialTools.utils.artificial_data import generate_random_relations


def test():
    ## Generate sp_relations
    sp_relations = generate_random_relations(100)
    connected = nx.connected_components(sp_relations.relations)

