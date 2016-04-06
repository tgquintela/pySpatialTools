
"""
test interpolation
------------------
Interpolators is a simple way to perform interpolation using this framework.

"""

import numpy as np
from pySpatialTools.Retrieve import KRetriever, CircRetriever
from pySpatialTools.utils.artificial_data import \
    random_transformed_space_points
from pySpatialTools.FeatureManagement.aux_descriptormodels import\
    characterizer_summer
from pySpatialTools.FeatureManagement.Interpolation_utils import\
    create_weighted_function, general_density_assignation

from pySpatialTools.FeatureManagement.descriptormodel import Interpolator

#
#def test():
#    n = 100
#    locs = np.random.random((n, 2))*100
#    locs1 = random_transformed_space_points(n/10, 2, None)*100
#    feats0 = np.random.random((n/10, 4))
#    ret = KRetriever(locs1, 4)
#
#    f_dens0, f_dens1 = 'weighted_count', 'weighted_avg'
#    pars_d = {}
#    f_weights = ['null', 'linear', 'Trapezoid', 'inverse_prop', 'inverse_prop',
#                 'exponential', 'exponential', 'gaussian', 'gaussian',
#                 'surgaussian', 'surgaussian', 'surgaussian', 'sigmoid',
#                 'sigmoid']
#
#    pars_w1 = {'max_r': 0.2, 'max_w': 1, 'min_w': 0}
#    pars_w2 = {'max_r': 0.2, 'r2': 0.4, 'max_w': 1, 'min_w': 0}
#
#    pars_w3 = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-8, 'rescale': True}
#    pars_w3b = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-8, 'rescale': True}
#    pars_w5 = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'S': None,
#               'rescale': True}
#    pars_w5b = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'S': None,
#                'rescale': False}
#    pars_w6 = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'S': 0.5,
#               'rescale': True}
#    pars_w7 = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'r_char': 0, 'B': None,
#               'rescale': True}
#    pars_w7b = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'r_char': 0, 'B': 2,
#                'rescale': False}
#    pars_w = [{}, pars_w1, pars_w2, pars_w3, pars_w3b, pars_w3, pars_w3b,
#              pars_w5, pars_w5b, pars_w5, pars_w5b, pars_w6, pars_w7, pars_w7b]
#
#    values, dists = np.random.random((10, 4)), np.random.random(10)
#
#    for i in range(len(f_weights)):
#        f = create_weighted_function(f_weights[i], pars_w[i], f_dens0, pars_d)
#        f(values, dists)
#        f = create_weighted_function(f_weights[i], pars_w[i], f_dens1, pars_d)
#        f(values, dists)
#        f = create_weighted_function(f_weights[i], pars_w[i],
#                                     characterizer_summer, pars_d)
#        f(values, dists)
#
#    M = general_density_assignation(locs, ret, np.ones(len(locs))*4, feats0,
#                                    f_weights[1], pars_w[1], f_dens1, pars_d)
#
#    Interpolator(f_weights[1], pars_w[1], f_dens1, pars_d)
