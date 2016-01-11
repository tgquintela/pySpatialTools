
"""
General function for computing spatial relation between spatial elements. This
module contains the general functions to compute the matrix or individual
relations.
"""

from itertools import combinations_with_replacement, product
import numpy as np


def general_spatial_relation(sp_el1, sp_el2, f):
    """General function for computing spatial relations with a function f
    given.

    Parameters
    ----------
    sp_el1:
    sp_el2:
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
    Sp_els:
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
    n = Sp_els.shape[0]
    if simmetry:
        pairs = combinations_with_replacement(xrange(n), 2)
    else:
        pairs = product(xrange(n), xrange(n))

    # 1. Compute the matrix of spatial relations
    Rel = np.zeros(n, n)
    for pair in pairs:
        rel = general_spatial_relation(Sp_els[pair[0]], Sp_els[pair[1]], f)
        Rel[pair[0], pair[1]], Rel[pair[1], pair[0]] = rel, rel

    return Rel
