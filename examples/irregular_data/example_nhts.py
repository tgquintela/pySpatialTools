
"""
Example nhts
------------
Time series with non homogenously time sampling.

* Interpolation
* Local measures
* Change of space sampling retrieve?

"""

import numpy as np
from scipy.interpolate import griddata
from pySpatialTools.Retrieve import KRetriever, CircRetriever
from pySpatialTools.FeatureManagement.features_objects import ImplicitFeatures
from pySpatialTools.FeatureManagement.Descriptors import Interpolator
from pySpatialTools.FeatureManagement import SpatialDescriptorModel
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager

import matplotlib.pyplot as plt


if __name__ == "__main__":
    ## Create parameters to initilize problem
    n_t = 1000
    # Generate time series information
    t_s = np.cumsum(np.multiply(np.random.random(n_t),
                    np.abs(np.random.normal(0.1, 0.5, size=n_t))))
    values = np.random.random(n_t) + np.cos(np.random.random(n_t)) +\
        np.cumsum(np.random.random(n_t)*np.random.normal(0, 0.2, size=n_t))

    # Interpolate to points
    t_int = np.arange(0, np.floor(t_s[-1]+0.5))
    nt_new = len(t_int)
    # Retrievers
    ret0 = KRetriever(locs=t_s.reshape((n_t, 1)), info_ret=1,
                      autolocs=t_int.reshape((len(t_int), 1)), ifdistance=True)
    ret1 = CircRetriever(locs=t_s.reshape((n_t, 1)), info_ret=.9,
                         autolocs=t_int.reshape((len(t_int), 1)),
                         ifdistance=True)

    ## Easy interpolation using scipy
    vals_nn = griddata(t_s.reshape((n_t, 1)), values.reshape((n_t, 1)),
                       t_int.reshape((len(t_int), 1)), fill_value=0.).ravel()

    ## Using pySpatialTools framework
    interpolator = Interpolator('null', {}, 'null', {})
    feats = ImplicitFeatures(values.reshape((len(values), 1)),
                             descriptormodel=interpolator, out_type='ndarray')
    feats_ret = FeaturesManager(feats, maps_vals_i=('matrix', nt_new, nt_new))
    interpolation = SpatialDescriptorModel(ret0, feats_ret)
    vals_nn_new = interpolation.compute().ravel()

    ## Plot interpolation
    plt.plot(t_s, values, label='original')
    plt.plot(range(nt_new), vals_nn, label='scipy interpolation')
    plt.plot(range(nt_new), vals_nn_new, label='pst 1-nn interpolation')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.show()

#    interpolator = Interpolator('gaussian', pars_w[1], f_dens1, pars_d)
