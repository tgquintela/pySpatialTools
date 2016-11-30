
"""
Density assignation
-------------------
Module to assign geographically density value to a points.

TODO
----
- Use neighbourhood defintion?
- Recurrent measure (TODO)[better before with the population?]

"""

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np


def general_density_assignation(locs, retriever, info_ret, values, f_weights,
                                params_w, f_dens, params_d):
    """General function for density assignation task.

    Parameters
    ----------
    locs: array_like shape(n, 2)
        location variables
    retriever: pySpatialTools.Retrieve.retrievers object
        retriever. Return the indices and distances of the possible retrivable
        points.
    values: array_like, shape (num. of retrievable candidates)
        values we will use to compute density.
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
    M: array_like, shape(n)
        mesasure of each location given.

    """

    ## 0. Preparation needed variables
    #parameters = preparation_parameters(parameters)
    if len(values.shape) == 1:
        values = values.reshape(values.shape[0], 1)

    ## 1. Computation of density
    M = compute_measure(locs, retriever, info_ret, values, f_weights, params_w,
                        f_dens, params_d)

    return M


###############################################################################
############################### Compute measure ###############################
###############################################################################
def compute_measure(locs, retriever, info_ret, values, f_weighs, params_w,
                    f_dens, params_d):
    """Function to compute assignation.

    Parameters
    ----------
    locs: array_like shape(n, 2)
        location variables
    retriever: pySpatialTools.Retrieve.retrievers object
        retriever. Return the indices and distances of the possible retrivable
        points.
    values: array_like, shape (num. of retrievable candidates)
        values we will use to compute density.
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
    M: array_like, shape(n)
        mesasure of each location given.

    """
    ## Computation of the measure based in the distances as weights.
    #M = np.zeros(locs.shape[0])
    M = []
    for i in xrange(locs.shape[0]):
        # Retrieve neighs_info
        neighs_info = retriever.retrieve_neighs(locs[[i]], info_ret[i], True)
        # Format neighs and dists
        neighs, dist, _, _ = neighs_info.get_information(k=0)
        neighs, dist = neighs[0][0], dist[0][0]
        neighs, dist = np.array(neighs).astype(int).ravel(), np.array(dist)
        # Get weights
        weights = from_distance_to_weights(dist, f_weighs, params_w)
        # Compute measure for i
        M_aux = compute_measure_i(weights, values[neighs], f_dens, params_d)
        M.append(M_aux)
        M = [0]
    M = np.array(M)
    return M


def compute_measure_i(weights, values, f_dens, params_d):
    """Swither function between different possible options to compute density.

    Parameters
    ----------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.
    values: array_like, shape (num. of retrievable candidates)
        values we will use to compute density.
    f_dens: function, set_scale_surgauss
        function of density assignation.
    params_d: dict
        parameters needed to apply f_dens.

    Returns
    -------
    measure: float
        the measure of assignation to the element with the neighbourhood
        described by the weights and values input.

    """
    if type(f_dens) == str:
        if f_dens == 'weighted_count':
            measure = compute_measure_wcount(weights, values, **params_d)
        elif f_dens == 'weighted_avg':
            measure = compute_measure_wavg(weights, values, **params_d)
        elif f_dens == 'null':
            measure = compute_measure_null(weights, values, **params_d)
    else:
        measure = f_dens(weights, values, **params_d)

    return measure


def compute_measure_wcount(weights, values):
    """Measure to compute density only based on the weighted count of selected
    elements around the point considered.

    Parameters
    ----------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.
    values: array_like, shape (num. of retrievable candidates)
        values we will use to compute density.

    Returns
    -------
    measure: float
        the measure of assignation to the element with the neighbourhood
        described by the weights and values input.

    """
    measure = np.sum(weights)
    return measure


def compute_measure_wavg(weights, values):
    """Measure to compute density based on the weighted average of selected
    elements around the point considered.

    Parameters
    ----------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.
    values: array_like, shape (num. of retrievable candidates)
        values we will use to compute density.

    Returns
    -------
    measure: float
        the measure of assignation to the element with the neighbourhood
        described by the weights and values input.

    """
#    measure = np.sum((np.array(weights) * np.array(values).T).T, axis=0)
    measure = np.dot(weights, values)
    return measure


def compute_measure_null(weights, values):
    """Null measure computation.

    Parameters
    ----------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.
    values: array_like, shape (num. of retrievable candidates)
        values we will use to compute density.

    Returns
    -------
    measure: float
        the measure of assignation to the element with the neighbourhood
        described by the weights and values input.

    """
    measure = values[0]
    return measure


# method, params (weitghted count, ...)
# method, params (linear, trapezoid,...)
###############################################################################
############################# Distance to weights #############################
###############################################################################
def from_distance_to_weights(dist, method, params):
    """Function which transforms the distance given to weights.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    method: str, optional
        the method we want to use in order to transform distances into weights.
    params: dict
        the paramters used to transform distance into weights.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """

    if type(method) == str:
        if method == 'linear':
            weights = dist2weights_linear(dist, **params)
        elif method == 'Trapezoid':
            weights = dist2weights_trapez(dist, **params)
        elif method == 'inverse_prop':
            weights = dist2weights_invers(dist, **params)
        elif method == 'exponential':
            weights = dist2weights_exp(dist, **params)
        elif method == 'gaussian':
            weights = dist2weights_gauss(dist, **params)
        elif method == 'surgaussian':
            weights = dist2weights_surgauss(dist, **params)
        elif method == 'sigmoid':
            weights = dist2weights_sigmoid(dist, **params)
        else:
            weights = dist
    else:
        weights = method(dist, **params)
    weights = np.array(weights).ravel()
    return weights


def dist2weights_linear(dist, max_r, max_w=1, min_w=0):
    """Linear distance weighting.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=0)
        minimum weight to be considered.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """
    weights = (max_w - dist)*((max_w-min_w)/float(max_r))+min_w
    return weights


def dist2weights_trapez(dist, max_r, r2, max_w=1, min_w=0):
    """Trapezoidal distance weighting.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    max_r: float
        maximum radius of the neighbourhood considered.
    r2: float
        intermediate radius.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=0)
        minimum weight to be considered.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """
    if type(dist) == np.ndarray:
        weights = dist2weights_linear(dist-r2, max_r-r2, max_w, min_w)
        weights[dist <= r2] = max_w
    else:
        if dist <= r2:
            weights = max_w
        else:
            weights = dist2weights_linear(dist-r2, max_r-r2, max_w, min_w)
    return weights


def dist2weights_invers(dist, max_r, max_w=1, min_w=1e-8, rescale=True):
    """Inverse distance weighting.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=1e-8)
        minimum weight to be considered.
    rescale: boolean (default=True)
        if re-scale the magnitude.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """
    if min_w == 0:
        tau = 1.
    else:
        tau = (max_w/min_w-1)/max_r
    if rescale:
        floor_f = 1./float(1.+tau*max_r)
        aux_dist = (1.+tau*dist).astype(float)
        weights = max_w/(1.-floor_f) * (1./aux_dist-floor_f)
    else:
        division = 1.+tau*dist
        if '__len__' in dir(division):
            division = division.astype(float)
        weights = np.divide(max_w, division)
    return weights


def dist2weights_exp(dist, max_r, max_w=1, min_w=1e-8, rescale=True):
    """Exponential distanve weighting.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=1e-8)
        minimum weight to be considered.
    rescale: boolean (default=True)
        if re-scale the magnitude.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """

    if min_w == 0:
        C = 1.
    else:
        C = -np.log(min_w/max_w)
    if rescale:
        weights = max_w/(1.-np.exp(-C)) * np.exp(-C*dist/max_r)
    else:
        weights = max_w * np.exp(-C*dist/max_r)
    return weights


def dist2weights_gauss(dist, max_r, max_w=1, min_w=1e-3, S=None, rescale=True):
    """Gaussian distance weighting.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=1e-8)
        minimum weight to be considered.
    S: float or None (default=None)
        the scale magnitude.
    rescale: boolean (default=True)
        if re-scale the magnitude.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """
    if S is None:
        S = set_scale_gauss(max_r, max_w, min_w)
    if rescale:
        A = max_w/(norm.pdf(0, scale=S)-norm.pdf(max_r, scale=S))
        weights = A*norm.pdf(dist, scale=S)
    else:
        A = max_w/norm.pdf(0, scale=S)
        weights = A*norm.pdf(dist, scale=S)
    return weights


def dist2weights_surgauss(dist, max_r, max_w=1, min_w=1e-3, S=None,
                          rescale=True):
    """Survival gaussian distance weighting.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=1e-8)
        minimum weight to be considered.
    S: float or None (default=None)
        the scale magnitude.
    rescale: boolean (default=True)
        if re-scale the magnitude.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """

    if S is None:
        S = set_scale_surgauss(max_r, max_w, min_w)
    if rescale:
        A = max_w/(norm.sf(0, scale=S)-norm.sf(max_r, scale=S))
        weights = A*(norm.sf(dist, scale=S)-norm.sf(max_r, scale=S))
    else:
        A = max_w/norm.sf(0)
        weights = A*norm.sf(dist, scale=S)
    return weights


def dist2weights_sigmoid(dist, max_r, max_w=1, min_w=1e-3, r_char=0, B=None,
                         rescale=True):
    """Sigmoid-like distance weighting.

    Parameters
    ----------
    dist: float or np.ndarray
        the distances to be transformed into weights.
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=1e-8)
        minimum weight to be considered.
    r_char: float (default=0)
        characteristic radius.
    B: float or None (default=None)
        a scale parameter.
    rescale: boolean (default=True)
        if re-scale the magnitude.

    Returns
    -------
    weights: np.ndarray, array_like, shape (num. of retrievable candidates)
        values of the weight of the neighs inferred from the distances.

    """

    C = r_char*max_r
    if B is None:
        B = set_scale_sigmoid(max_r, max_w, min_w, r_char)
    sigmoid = lambda x: 1./(1.+B*np.exp(x+C))
    if rescale:
        floor_f = sigmoid(max_r)
        weights = max_w/(sigmoid(0)-floor_f)*(sigmoid(dist)-floor_f)
    else:
        weights = 1./(1.+B*np.exp(dist+C))
    return weights


###############################################################################
############################# Set scale functions #############################
###############################################################################
def set_scales_kernel(method, max_r, max_w, min_w, r_char=0):
    """Switcher function for set scale functions.

    Parameters
    ----------
    method: str, optional ['surgaussian', 'gaussian', 'sigmoid']
        the method used to set the scales kernel.
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int (default=1)
        maximum weight to be considered.
    min_w: float (default=1e-8)
        minimum weight to be considered.
    r_char: float (default=0)
        characteristic radius.

    Returns
    -------
    scale: float
        scale value.

    """
    if method == 'surgaussian':
        scale = set_scale_surgauss(max_r, max_w, min_w)
    elif method == 'gaussian':
        scale = set_scale_gauss(max_r, max_w, min_w)
    elif method == 'sigmoid':
        scale = set_scale_sigmoid(max_r, max_w, min_w, r_char)
    return scale


def set_scale_surgauss(max_r, max_w, min_w):
    """Set the scale factor of the surgauss kernel.

    Parameters
    ----------
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int
        maximum weight to be considered.
    min_w: float
        minimum weight to be considered.

    Returns
    -------
    scale: float
        scale value.

    """
    A = max_w/norm.sf(0)
    f_err = lambda x: (A*norm.sf(max_r, scale=x)-min_w)**2
    scale = minimize(f_err, x0=np.array([max_r]), method='Powell', tol=1e-8)
    scale = float(scale['x'])
    return scale


def set_scale_gauss(max_r, max_w, min_w):
    """Set the scale factor of the gauss kernel.

    Parameters
    ----------
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int
        maximum weight to be considered.
    min_w: float
        minimum weight to be considered.

    Returns
    -------
    scale: float
        scale value.

    """
    A = max_w/norm.pdf(0)
    f_err = lambda x: (A*norm.pdf(max_r, scale=x)-min_w)**2
    scale = minimize(f_err, x0=np.array([0]), method='Powell', tol=1e-8)
    scale = float(scale['x'])
    return scale


def set_scale_sigmoid(max_r, max_w, min_w, r_char):
    """Set scale for sigmoidal functions.

    Parameters
    ----------
    max_r: float
        maximum radius of the neighbourhood considered.
    max_w: int
        maximum weight to be considered.
    min_w: float
        minimum weight to be considered.
    r_char: float
        characteristic radius.

    Returns
    -------
    scale: float
        scale value.

    """
    C = r_char*max_r
    sigmoid_c = lambda B: (1./(1.+B*np.exp(max_r+C)) - min_w)**2
    B = minimize(sigmoid_c, x0=np.array([1]), method='BFGS',
                 tol=1e-8, bounds=(0, None))
    B = B['x'][0]
    return B


###############################################################################
############################# Preparation inputs #############################
###############################################################################
#def preparation_parameters(parameters):
#    "Function to put into coherence the selected parameters."
#
#    method = parameters['params']['method']
#    params = parameters['params']['params']
#    if method == 'gaussian':
#        bool_scale = 'S' in params
#        if not bool_scale:
#            scale = set_scale_gauss(params['max_r'], params['max_w'],
#                                    params['min_w'])
#            parameters['params']['params']['S'] = scale
#    elif method == 'surgaussian':
#        bool_scale = 'S' in params
#        if not bool_scale:
#            scale = set_scale_surgauss(params['max_r'], params['max_w'],
#                                       params['min_w'])
#            parameters['params']['params']['S'] = scale
#    elif method == 'sigmoid':
#        bool_scale = 'B' in params
#        if not bool_scale:
#            scale = set_scale_sigmoid(params['max_r'], params['max_w'],
#                                      params['min_w'], params['r_char'])
#            parameters['params']['params']['B'] = scale
#
#    return parameters
