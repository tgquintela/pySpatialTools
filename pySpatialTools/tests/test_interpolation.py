
"""
test interpolation
------------------
Interpolators is a simple way to perform interpolation using this framework.

"""

import os
import numpy as np

from pySpatialTools.Retrieve import KRetriever, CircRetriever
from pySpatialTools.utils.artificial_data import \
    random_transformed_space_points
from pySpatialTools.FeatureManagement.aux_descriptormodels import\
    characterizer_summer
from pySpatialTools.FeatureManagement.Interpolation_utils import\
    create_weighted_function, general_density_assignation,\
    DensityAssign_Process
from pySpatialTools.FeatureManagement.Interpolation_utils.\
    density_assignation import set_scales_kernel
from pySpatialTools.FeatureManagement.Interpolation_utils.\
    density_utils import clustering_by_comparison


from pySpatialTools.FeatureManagement.descriptormodel import Interpolator
from pySpatialTools.utils.util_external import Logger


def test():
    n = 100
    locs = np.random.random((n, 2))*100
    locs1 = random_transformed_space_points(n/10, 2, None)*100
    feats0 = np.random.random((n/10, 4))
    feats1 = np.random.random(n/10)
    kneighs4 = np.ones(len(locs)).astype(int)*4
    ret = KRetriever(locs1, 4, ifdistance=True)
    ret1 = KRetriever(locs1, ifdistance=True)

    f_dens0, f_dens1 = 'weighted_count', 'weighted_avg'
    pars_d = {}
    f_weights = ['null', 'linear', 'Trapezoid', 'inverse_prop', 'inverse_prop',
                 'exponential', 'exponential', 'gaussian', 'gaussian',
                 'surgaussian', 'surgaussian', 'surgaussian', 'sigmoid',
                 'sigmoid']

    pars_w1 = {'max_r': 0.2, 'max_w': 1, 'min_w': 0}
    pars_w2 = {'max_r': 0.2, 'r2': 0.4, 'max_w': 1, 'min_w': 0}

    pars_w3 = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-8, 'rescale': True}
    pars_w3b = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-8, 'rescale': False}
    pars_w5 = {'max_r': 0.2, 'max_w': 1, 'min_w': 0, 'S': None,
               'rescale': True}
    pars_w5b = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'S': None,
                'rescale': False}
    pars_w6 = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'S': 0.5,
               'rescale': True}
    pars_w7 = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'r_char': 0, 'B': None,
               'rescale': True}
    pars_w7b = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'r_char': 0, 'B': 2,
                'rescale': False}
    pars_w = [{}, pars_w1, pars_w2, pars_w3, pars_w3b, pars_w3, pars_w3b,
              pars_w5, pars_w5b, pars_w5, pars_w5b, pars_w6, pars_w7, pars_w7b]

    values, dists = np.random.random((10, 4)), np.random.random(10)

    pars_now = {'max_r': 0.2, 'max_w': 1, 'min_w': 1e-3, 'r_char': 0}
    set_scales_kernel('surgaussian', **pars_now)
    set_scales_kernel('gaussian', **pars_now)
    set_scales_kernel('sigmoid', **pars_now)

    for i in range(len(f_weights)):
        f = create_weighted_function(f_weights[i], pars_w[i], f_dens0, pars_d)
        f(values, dists)
        f = create_weighted_function(f_weights[i], pars_w[i], f_dens1, pars_d)
        f(values, dists)
        f = create_weighted_function(f_weights[i], pars_w[i],
                                     characterizer_summer, pars_d)
        f(values, dists)

    M = general_density_assignation(locs, ret, kneighs4, feats0,
                                    f_weights[1], pars_w[1], f_dens1, pars_d)
    M = general_density_assignation(locs, ret, kneighs4, feats1,
                                    f_weights[1], pars_w[1], f_dens1, pars_d)
    M = general_density_assignation(locs, ret1, kneighs4, feats1,
                                    f_weights[1], pars_w[1], f_dens1, pars_d)

    interpolator = Interpolator(f_weights[1], pars_w[1], f_dens1, pars_d)

    ###########################################################################
    ###########################################################################
    #### Density assignation testing
    ################################
    logfile = Logger('logfile.log')
    pars_dens_asign = {'f_weights': f_weights[1], 'params_w': pars_w[1],
                       'f_dens': f_dens1, 'params_d': pars_d}
    locs_data = locs
    pop_data = feats1
    retriever = KRetriever
    dens_asign = DensityAssign_Process(logfile, retriever)
    dens_asign.compute_density(locs1, locs_data, pop_data, kneighs4,
                               pars_dens_asign)
    os.remove('logfile.log')

    ###########################################################################
    ###########################################################################
    #### Interpolation utils testing
    ################################
    density0 = np.random.random(50)*2
    density1 = np.random.random(50)*2
    clustering_by_comparison(density0, density1)
