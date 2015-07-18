
import numpy as np

from pySpatialTools.Retrieve import general_density_assignation
from pySpatialTools.Retrieve import CircRetriever
from pySpatialTools.Retrieve.density_assignation import dist2weights_exp, compute_measure_wavg

n, r = 100000, .001

f_weigh = dist2weights_exp
params_w = {'max_r': 0.1}
f_dens = compute_measure_wavg
params_d = {}
info_ret = np.ones(n)*r

locs = np.random.random((n, 2))
values = np.ones(n)
retriever = CircRetriever(locs, False)

M = general_density_assignation(locs, retriever, info_ret, values, f_weigh, params_w, f_dens, params_d)


