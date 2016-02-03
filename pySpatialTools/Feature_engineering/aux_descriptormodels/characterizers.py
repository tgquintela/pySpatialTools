
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
from pySpatialTools.Feature_engineering.Interpolation_utils.\
    density_assignation import from_distance_to_weights, compute_measure_i


def characterizer_1sh_counter(pointfeats, point_pos):
    """Characterizer which counts the different types of elements in the
    neighbourhood of the element studied."""
    pointfeats = np.array(pointfeats).astype(int)
    descriptors = dict(Counter(pointfeats.ravel()))
    print '0', descriptors
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


def create_weighted_function(f_weighs, params_w, f_dens, params_d):
    """Functions which acts in order to create a weighted function. You have to
    give the needed parameters and it returns to you a function.

    Parameters
    ----------
    f_weighs: functions, str
        function of weighs assignation. It transforms the distance to weights.
    params_w: dict
        parameters needed to apply f_weighs.
    f_dens: function, set_scale_surgauss
        function of density assignation.
    params_d: dict
        parameters needed to apply f_dens.

    Returns
    -------
    f: function
        a function which has as inputs the dists and values of the neighs
        points.

    """

    def f(values, dists):
        weights = from_distance_to_weights(dists, f_weighs, params_w)
        M = compute_measure_i(weights, values, f_dens, params_d)
        return M

    return f
