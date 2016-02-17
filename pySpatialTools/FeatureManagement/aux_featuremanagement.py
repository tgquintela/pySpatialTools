
"""
Auxiliar featuremanagement
--------------------------
Auxiliar functions for management of features.

"""

import numpy as np
from features_objects import ExplicitFeatures


def create_aggfeatures(discretization, regmetric, features, descriptormodel):
    """Create aggregated features.
    """
    ## 00. Descomposing descriptormodel
    characterizer = lambda x, d: descriptormodel.characterizer(x, d)
    _nullvalue = descriptormodel._nullvalue
    agg_f = lambda x, d: descriptormodel.reducer(x, d)

    ## 0. Preparing the inputs
    if type(discretization) == tuple:
        locs, discretizor = discretization
        regs = discretizor.discretize(locs)
    else:
        regs = discretization
    u_regs = np.unique(regs)
    u_regs = u_regs.reshape((len(u_regs), 1))

    ## 1. Compute aggregation
    sh = features.shape
    agg = np.ones((len(u_regs), sh[1], sh[2])) * _nullvalue
    for i in xrange(len(u_regs)):
        neighs_info = regmetric.retrieve_neighs(u_regs[i])
        if list(neighs_info[0]) != []:
            for k in range(sh[2]):
                agg[i, :, k] = agg_f(features[neighs_info, k], neighs_info[1])
        else:
            agg[i, :, :] = np.ones((sh[1], sh[2])) * _nullvalue

    ## 2. Prepare output
    agg = ExplicitFeatures(agg, indices=u_regs, characterizer=characterizer)
    ## TODO: names=[], nullvalue=None

    return agg


def compute_featuresnames(descriptormodel, featureobject):
    """Compute the possible feature names from the pointfeatures."""
    if type(featureobject) == np.ndarray:
        featuresnames = descriptormodel._f_default_names(featureobject)
        return featuresnames
    if 'typefeat' in dir(featureobject):
        featuresnames =\
            descriptormodel._f_default_names(featureobject.features)
    return featuresnames
