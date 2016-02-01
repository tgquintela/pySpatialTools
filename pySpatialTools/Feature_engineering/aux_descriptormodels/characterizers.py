
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

"""

import numpy as np
from collections import Counter


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
    return descriptors


def characterizer_average(pointfeats, point_pos):
    """Characterizer which average the point features of elements in the
    neighbourhood of the element studied."""
    pointfeats = np.array(pointfeats)
    descriptors = np.mean(pointfeats, axis=0)
    return descriptors
