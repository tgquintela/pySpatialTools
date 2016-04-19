
"""
artificial data
----------------
functions which creates artificial data.
"""

import numpy as np
from pySpatialTools.Discretization import SetDiscretization


def random_membership(n_elements, n_collections, multiple=True):
    """Function to create membership relations for set discretizor."""
    if multiple:
        membership = list_membership(n_elements, n_collections)
    else:
        membership = np.random.randint(0, n_collections, n_elements)
    set_disc = SetDiscretization(membership)
    return set_disc


def list_membership(n_elements, n_collections):
    membership = []
    for i in xrange(n_elements):
        aux = np.random.randint(0, n_collections,
                                np.random.randint(2*n_collections))
        aux = np.unique(aux)
        membership.append(aux)
    return membership
