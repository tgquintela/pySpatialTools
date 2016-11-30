
"""
utils spatial relations
-----------------------
Module which contains utils to compute spatial relations between spatial
elements.

"""

import numpy as np
from itertools import combinations_with_replacement, product


def general_spatial_relation(sp_el1, sp_el2, f):
    """General function for computing spatial relations with a function f
    given.

    Parameters
    ----------
    sp_el1: optional
        the spatial information of element 1.
    sp_el2: optional
        the spatial information of element 2.
    f: function
        function to compute spatial relation between spatial objects.

    Returns
    -------
    rel: float
        number of the the relation between the spatial object.

    """
    rel = f(sp_el1, sp_el2)
    return rel


def general_spatial_relations(Sp_els, f, simmetry=False):
    """General function for computing the spatial relations between each
    elements of the collection Sp_els.

    Parameters
    ----------
    Sp_els: array_like
        the spatial elements collection.
    f: function
        function to compute spatial relation between spatial objects.
    simmetry: boolean
        if the Rel measure is simmetry.

    Returns
    -------
    Rel: np.ndarray
        the relation matrix.

    """

    # 0. Compute previous variables
    n = len(Sp_els)
    if simmetry:
        pairs = combinations_with_replacement(xrange(n), 2)
    else:
        pairs = product(xrange(n), xrange(n))

    # 1. Compute the matrix of spatial relations
    Rel = np.zeros((n, n))
    for pair in pairs:
        rel = general_spatial_relation(Sp_els[pair[0]], Sp_els[pair[1]], f)
        Rel[pair[0], pair[1]], Rel[pair[1], pair[0]] = rel, rel

    return Rel
