
"""
artificial data
----------------
Functions which creates artificial data.
"""

import numpy as np
from pySpatialTools.Discretization import SetDiscretization


def random_membership(n_elements, n_collections, multiple=True):
    """Function to create membership relations for set discretizor.

    Parameters
    ----------
    n_elements: int
        the number of elements.
    n_collections: int
        the number of collections.
    multiple: boolean (default=True)
        if we want to have multiple collections associated to an individual
        element.

    Returns
    -------
    set_disc: pst.Discretization.SetDiscretization
        the membership information stored in the set discretization way.

    """
    if multiple:
        membership = list_membership(n_elements, n_collections)
    else:
        membership = np.random.randint(0, n_collections, n_elements)
    set_disc = SetDiscretization(membership)
    return set_disc


def list_membership(n_elements, n_collections):
    """Generation of list membership.

    Parameters
    ----------
    n_elements: int
        the number of elements.
    n_collections: int
        the number of collections.

    Returns
    -------
    membership: list
        the membership relations information.

    """
    membership = []
    for i in xrange(n_elements):
        aux = np.random.randint(0, n_collections,
                                np.random.randint(2*n_collections))
        aux = np.unique(aux)
        membership.append(aux)
    return membership
