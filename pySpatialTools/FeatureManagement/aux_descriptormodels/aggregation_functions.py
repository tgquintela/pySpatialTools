
"""
aggregation functions
---------------------
This module contains aggregation functions to be used by the aggregation
function of the features object.

Standart inputs for the aggregation:
------------------------------------
    Parameters
    ----------
    pointfeats: np.ndarray or list
        the indices are (idxs, neighs, nfeats) or [idxs][neighs]{feats}
    point_pos: list
        the indices are [idxs][neighs]

    Returns
    -------
    descriptors: np.ndarray or list
        the result is expressed with the indices (idxs, nfeats)
        [idxs]{nfeats}

"""

#import numpy as np
from characterizers import characterizer_1sh_counter, characterizer_summer,\
    characterizer_average


def aggregator_1sh_counter(pointfeats, point_pos):
    """Aggregator which counts the different types of elements in the
    aggregation units.

    Parameters
    ----------
    pointfeats: np.ndarray or list
        the indices are (idxs, neighs, nfeats) or [idxs][neighs]{feats}
    point_pos: list
        the indices are [idxs][neighs]

    Returns
    -------
    descriptors: np.ndarray or list
        the result is expressed with the indices (idxs, nfeats)
        [idxs]{nfeats}

    """
    descriptors = characterizer_1sh_counter(pointfeats, point_pos)
    return descriptors


def aggregator_summer(pointfeats, point_pos):
    """Aggregator which sums the different element features in the
    aggregation units.

    Parameters
    ----------
    pointfeats: np.ndarray or list
        the indices are (idxs, neighs, nfeats) or [idxs][neighs]{feats}
    point_pos: list
        the indices are [idxs][neighs]

    Returns
    -------
    descriptors: np.ndarray or list
        the result is expressed with the indices (idxs, nfeats)
        [idxs]{nfeats}

    """
    descriptors = characterizer_summer(pointfeats, point_pos)
    return descriptors


def aggregator_average(pointfeats, point_pos):
    """Aggregator which average the different element features in the
    aggregation units.

    Parameters
    ----------
    pointfeats: np.ndarray or list
        the indices are (idxs, neighs, nfeats) or [idxs][neighs]{feats}
    point_pos: list
        the indices are [idxs][neighs]

    Returns
    -------
    descriptors: np.ndarray or list
        the result is expressed with the indices (idxs, nfeats)
        [idxs]{nfeats}

    """
    descriptors = characterizer_average(pointfeats, point_pos)
    return descriptors
