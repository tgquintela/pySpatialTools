
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
"""

import numpy as np
from collections import Counter
from ..Interpolation_utils.density_assignation import\
    from_distance_to_weights, compute_measure_i


def characterizer_1sh_counter(pointfeats, point_pos):
    """Characterizer which counts the different types of elements in the
    neighbourhood of the element studied."""
    pointfeats = np.array(pointfeats).astype(int)
    descriptors = dict(Counter(pointfeats.ravel()))
    return descriptors


#def characterizer_nsh_counter(pointfeats, point_pos):
#    "Compulsary function to pass for the feture retriever."
#    pointfeats = np.array(pointfeats)
#    descriptors = np.sum(pointfeats, axis=0)
#    return descriptors


def characterizer_summer(pointfeats, point_pos):
    """Characterizer which sums the point features of elements in the
    neighbourhood of the element studied."""
    pointfeats = np.array(pointfeats)
    descriptors = np.sum(pointfeats, axis=0)
    ## 2-dim shape
    new_sh = tuple([1] + list(descriptors.shape))
    descriptors = descriptors.reshape(new_sh)
    return descriptors


def characterizer_average(pointfeats, point_pos):
    """Characterizer which average the point features of elements in the
    neighbourhood of the element studied."""
    pointfeats = np.array(pointfeats)
    descriptors = np.mean(pointfeats, axis=0)
    ## 2-dim shape
    new_sh = tuple([1] + list(descriptors.shape))
    descriptors = descriptors.reshape(new_sh)
    return descriptors
