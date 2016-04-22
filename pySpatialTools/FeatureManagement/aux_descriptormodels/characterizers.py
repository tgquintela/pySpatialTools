
"""
Characterizers
--------------
Module which contains different useful characterizer functions using partial
information.
This function is a compulsary function in the descriptor model object in
order to be passed to the feture retriever.

** Main properties:
   ---------------
INPUTS:
- pointfeats: the features associated directly to each element.
- point_pos: relative position to the element neighbourhood.

OUTPUTS:
- descriptors: in dict or array format depending on the descriptors convenience

WARNING: output requires TWO dimensions

TODO
----
Dict-based features.

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

import numpy as np
from collections import Counter
#from itertools import product
from ..Interpolation_utils.density_assignation import\
    from_distance_to_weights, compute_measure_i


############################## Container function #############################
###############################################################################
def characterizer_from_unitcharacterizer(f):
    """Transform a common individual characterizer to a standard characterizer
    function."""
    def new_characterizer(pointfeats, point_pos):
        """New characterizer from an individual one."""
        descriptors = []
        for i in range(len(pointfeats)):
            descriptors.append(f(pointfeats[i], point_pos[i]))
        ## Assumption of uniform type output
        if type(descriptors[0]) == np.ndarray:
            descriptors = np.array(descriptors)
        return descriptors
    return new_characterizer


################################### Counter ###################################
###############################################################################
def characterizer_1sh_counter(pointfeats, point_pos):
    """Characterizer which counts the different types of elements in the
    neighbourhood of the element studied.
    Inputs standards:
    * [iss_i](nei, feats)
    * (iss_i, nei, feats)
    n_feats = 1
    """
    n_iss = len(pointfeats)
    descriptors = [[]]*n_iss
    for i in range(n_iss):
        pointfeats_i = np.array(pointfeats[i]).astype(int).ravel()
        descriptors[i] = dict(Counter(pointfeats_i))
    return descriptors


################################### Summer ####################################
###############################################################################
def characterizer_summer(pointfeats, point_pos):
    """Characterizer which sums the point features of elements in the
    neighbourhood of the element studied.
    Inputs standards:
    * [iss_i](nei, feats)
    * (iss_i, nei, feats)
    * (nei, feats)
    * [iss_i][nei]{feats}
    """
    if type(pointfeats) == np.ndarray:
        descriptors = characterizer_summer_array(pointfeats, point_pos)
    elif type(pointfeats) == list:
        if type(pointfeats[0]) == np.ndarray:
            descriptors = characterizer_summer_listarray(pointfeats,
                                                         point_pos)
        elif type(pointfeats[0][0]) == dict:
            descriptors = characterizer_summer_listdict(pointfeats,
                                                        point_pos)
    return descriptors


def characterizer_summer_array(pointfeats, point_pos):
    """Characterizer which sums the point features of elements in the
    neighbourhood of the element studied.
    Inputs standards:
    * [iss_i](nei, feats)
    * (iss_i, nei, feats)
    * (nei, feats)
    """
    if type(pointfeats) == list:
        descriptors = characterizer_summer_listarray(pointfeats, point_pos)
    else:
        descriptors = characterizer_summer_arrayarray(pointfeats, point_pos)
    return descriptors


def characterizer_summer_listdict(pointfeats, point_pos):
    """Characterizer which sums the point features of elements in the
    neighbourhood of the element studied.
    Inputs standards:
    * [iss_i][nei]{feats}
    """
    descriptors = []
    for i in range(len(pointfeats)):
        vals = []
        for nei in range(len(pointfeats[i])):
            vals += pointfeats[i][nei].keys()
        vals = list(set(vals))
        desc_i = {}
        for v in vals:
            value = [pointfeats[i][nei][v] for nei in range(len(pointfeats[i]))
                     if v in pointfeats[i][nei]]
            desc_i[v] = np.sum(value)
        descriptors.append(desc_i)
    return descriptors


def characterizer_summer_listarray(pointfeats, point_pos):
    descriptors = []
    for i in range(len(pointfeats)):
        sh = pointfeats[i].shape
        descriptors.append(np.sum(np.array(pointfeats[i]), axis=len(sh)-2))
    descriptors = np.array(descriptors)
    return descriptors


def characterizer_summer_arrayarray(pointfeats, point_pos):
    sh = pointfeats.shape
    pointfeats = np.array(pointfeats)
    descriptors = np.sum(pointfeats, axis=len(sh)-2)
    return descriptors


################################### Average ###################################
###############################################################################
def characterizer_average(pointfeats, point_pos):
    """Characterizer which average the point features of elements in the
    neighbourhood of the element studied.
    Inputs standards:
    * [iss_i](nei, feats)
    * (iss_i, nei, feats)
    * (nei, feats)
    * [iss_i][nei]{feats}
    """
    if type(pointfeats) == np.ndarray:
        descriptors = characterizer_average_arrayarray(pointfeats, point_pos)
    elif type(pointfeats) == list:
        if type(pointfeats[0]) == np.ndarray:
            descriptors = characterizer_average_listarray(pointfeats,
                                                          point_pos)
        elif type(pointfeats[0][0]) == dict:
            descriptors = characterizer_average_listdict(pointfeats,
                                                         point_pos)
    return descriptors


def characterizer_average_array(pointfeats, point_pos):
    """Characterizer which average the point features of elements in the
    neighbourhood of the element studied.
    Inputs standards:
    * [iss_i](nei, feats)
    * (iss_i, nei, feats)
    * (nei, feats)
    """
    if type(pointfeats) == list:
        descriptors = characterizer_average_listarray(pointfeats, point_pos)
    else:
        descriptors = characterizer_average_arrayarray(pointfeats, point_pos)
    return descriptors


def characterizer_average_listdict(pointfeats, point_pos):
    """Characterizer which average the point features of elements in the
    neighbourhood of the element studied.
    Inputs standards:
    * [iss_i][nei]{feats}
    """
    descriptors = []
    for i in range(len(pointfeats)):
        vals = []
        for nei in range(len(pointfeats[i])):
            vals += pointfeats[i][nei].keys()
        vals = list(set(vals))
        desc_i = {}
        for v in vals:
            value = [pointfeats[i][nei][v] for nei in range(len(pointfeats[i]))
                     if v in pointfeats[i][nei]]
            desc_i[v] = np.mean(value)
        descriptors.append(desc_i)
    return descriptors


def characterizer_average_listarray(pointfeats, point_pos):
    descriptors = []
    for i in range(len(pointfeats)):
        sh = pointfeats[i].shape
        descriptors.append(np.sum(np.array(pointfeats[i]), axis=len(sh)-2))
    descriptors = np.array(descriptors)
    return descriptors


def characterizer_average_arrayarray(pointfeats, point_pos):
    sh = pointfeats.shape
    pointfeats = np.array(pointfeats)
    descriptors = np.mean(pointfeats, axis=len(sh)-2)
    return descriptors
