
"""
Assign geographically density value to a points.
"""

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np


def general_density_assignation(locs, parameters, values=None, locs2=None):
    "Density assignation function."

    # Creation of the kdtree for retrieving neighs
    if locs2 is None:
        leafsize = int(locs.shape[0]/float(10.))
        kdtree = KDTree(locs, leafsize=leafsize)
    else:
        leafsize = int(locs2.shape[0]/float(10.))
        kdtree = KDTree(locs2, leafsize=leafsize)

    parameters = preparation_parameters(parameters)

    M = compute_measure(locs=locs, kdtree=kdtree, values=values, **parameters)

    ## Recurrent measure (TODO)[better before with the population?]
    return M


# method, params (weitghted count, ...)
# method, params (linear, trapezoid,...)

###############################################################################
############################### Compute measure ###############################
###############################################################################
def compute_measure(locs, kdtree, max_r, values, method, params):
    "Retrieve the neighs indices and the neighs descriptors and weights."

    ## Computation of the measure based in the distances as weights.
    M = np.zeros(locs.shape[0])
    for i in range(locs):
        neighs, dist = get_self_neighs_i(locs, kdtree, max_r, i)
        M[i] = compute_measure_i(neighs, dist, values[neighs], method, params)
    return M


def get_self_neighs_i(locs, kdtree, max_r, i):
    "Retrieving neighs and distance."
    loc = locs[i, :]
    neighs = kdtree.query_ball_point(loc, max_r)
    neighs.remove(i)
    dist = cdist(locs[i, :], locs[neighs, :])
    return neighs, dist


def compute_measure_i(neighs, dist, values, method, params):
    "Swither function between different possible options to compute density."
    if method == 'weighted_count':
        measure = compute_measure_wcount(neighs, dist, params)
    elif method == 'weighted_avg':
        measure = compute_measure_wavg(neighs, dist, params)

    return measure


def compute_measure_wcount(neighs, dist, params):
    """Measure to compute density only based on the weighted count of selected
    elements around the point considered.
    """
    weights = from_distance_to_weights(dist, **params)
    measure = np.sum(weights)
    return measure


def compute_measure_wavg(neighs, dist, values, params):
    """Measure to compute density based on the weighted average of selected
    elements around the point considered.
    """
    weights = from_distance_to_weights(dist, **params)
    measure = np.sum(weights * values)
    return measure


###############################################################################
############################# Distance to weights #############################
###############################################################################
def from_distance_to_weights(dist, method, params):
    "Function which transforms the distance given to weights."

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

    return weights


def dist2weights_linear(dist, max_r, max_w=1, min_w=0):
    "Linear distance weighting."
    weights = (max_w - dist)*((max_w-min_w)/float(max_r))+min_w
    return weights


def dist2weights_trapez(dist, max_r, r2, max_w=1, min_w=0):
    "Trapezoidal distance weighting."
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
    "Inverse distance weighting."
    if min_w == 0:
        tau = 1.
    else:
        tau = (max_w/min_w-1)/max_r
    if rescale:
        floor_f = 1./float(1.+tau*max_r)
        weights = max_w/(1.-floor_f) * (1./float(1.+tau*dist)-floor_f)
    else:
        weights = max_w/float(1.+tau*dist)
    return weights


def dist2weights_exp(dist, max_r, max_w=1, min_w=1e-8, rescale=True):
    "Exponential distanve weighting."
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
    "Gaussian distance weighting."
    if S is None:
        S = set_scale_surgauss(max_r, max_w, min_w)
    if rescale:
        A = max_w/(norm.pdf(0)-norm.pdf(max_r, scale=S))
        weights = A*norm.pdf(dist, scale=S)
    else:
        A = max_w/norm.pdf(0)
        weights = A*norm.pdf(dist, scale=S)
    return weights


def dist2weights_surgauss(dist, max_r, max_w=1, min_w=1e-3, S=None,
                          rescale=True):
    "Survival gaussian distance weighting."

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
    "Sigmoid-like distance weighting"

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
def set_scales_kernel(method, max_r, max_w, min_w, r_char=None):
    "Switcher function for set scale functions."
    if method == 'surgaussian':
        scale = set_scale_surgauss(max_r, max_w, min_w)
    elif method == 'gaussian':
        scale = set_scale_gauss(max_r, max_w, min_w)
    elif method == 'sigmoid':
        scale = set_scale_sigmoid(max_r, max_w, min_w, r_char)
    return scale


def set_scale_surgauss(max_r, max_w, min_w):
    "Set the scale factor of the surgauss kernel."
    A = max_w/norm.sf(0)
    scale = minimize(lambda x: (A*norm.sf(max_r, scale=x)-min_w)**2,
                     x0=np.array([max_r]), method='BFGS',
                     tol=1e-8, bounds=(0, None))
    scale = scale['x'][0]
    return scale


def set_scale_gauss(max_r, max_w, min_w):
    "Set the scale factor of the gauss kernel."
    A = max_w/norm.pdf(0)
    scale = minimize(lambda x: (A*norm.pdf(max_r, scale=x)-min_w)**2,
                     x0=np.array([max_r]), method='BFGS',
                     tol=1e-8, bounds=(0, None))
    scale = scale['x'][0]
    return scale


def set_scale_sigmoid(max_r, max_w, min_w, r_char):
    "Set scale for sigmoidal functions."
    C = r_char*max_r
    sigmoid_c = lambda B: 1./(1.+B*np.exp(max_r+C)) - min_w
    B = minimize((sigmoid_c)**2,
                 x0=np.array([1]), method='BFGS',
                 tol=1e-8, bounds=(0, None))
    return B


###############################################################################
############################# Preparation inputs #############################
###############################################################################
def preparation_parameters(parameters):
    "Function to put into coherence the selected parameters."

    method = parameters['params']['method']
    params = parameters['params']['params']
    if method == 'gaussian':
        bool_scale = 'S' in params
        if not bool_scale:
            scale = set_scale_gauss(params['max_r'], params['max_w'],
                                    params['min_w'])
            parameters['params']['params']['S'] = scale
    elif method == 'surgaussian':
        bool_scale = 'S' in params
        if not bool_scale:
            scale = set_scale_surgauss(params['max_r'], params['max_w'],
                                       params['min_w'])
            parameters['params']['params']['S'] = scale
    elif method == 'sigmoid':
        bool_scale = 'B' in params
        if not bool_scale:
            scale = set_scale_sigmoid(params['max_r'], params['max_w'],
                                      params['min_w'], params['r_char'])
            parameters['params']['params']['B'] = scale

    return parameters
