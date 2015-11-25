
"""
Weighting functions
-------------------
Module which depends on density_assignation module and creates some util
functions for the computation of weighted values.

"""

from density_assignation import from_distance_to_weights, compute_measure_i


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
        a function which has as inputs the dist and values of the neighs
        points.

    """

    def f(dist, values):
        weights = from_distance_to_weights(dist, f_weighs, params_w)
        M = compute_measure_i(weights, values, f_dens, params_d)
        return M

    return f
