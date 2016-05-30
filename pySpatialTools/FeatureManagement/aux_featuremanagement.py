
"""
Auxiliar featuremanagement
--------------------------
Auxiliar functions for management of features.

"""

import numpy as np
from features_objects import _featuresobject_parsing_creation
from features_retriever import _features_parsing_creation


###############################################################################
################### Candidates to aggregating_in functions ####################
###############################################################################
def dummy_aggregation_in(locs, regs, disc, descriptormodel, features,
                         pars_feat):

        #### aggfeatures, names=[], _nullvalue=0, characterizer=None
        new_features, new_descriptor,  = ''


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
    u_regs = np.array([u_regs[i] for i in range(len(u_regs))
                       if u_regs[i] in regmetric.data_input])
    u_regs = u_regs.reshape((len(u_regs), 1))

    ## 1. Compute aggregation
    # Prepare container of neighs information
    nei_nfo = regmetric.export_neighs_info()
    nei_nfo.reset_structure('tuple_tuple')
    nei_nfo.reset_level(2)
    # Prepare features
    sh = features.shape
    agg = np.ones((len(u_regs), sh[1], sh[2])) * _nullvalue
    for i in xrange(len(u_regs)):
        ## We get neighs_info with only 1-k and 1-iss
        neighs_info = regmetric.retrieve_neighs(u_regs[i])
        for k in neighs_info.ks:
            # Get neighs information
            neighs, dists, _, _ = neighs_info.get_information(k)
            print neighs_info.get_neighs, neighs_info.get_sp_rel_pos
            print nei_nfo.get_neighs, nei_nfo.get_sp_rel_pos
            print nei_nfo.set_neighs, nei_nfo.set_sp_rel_pos
            neighs, dists = neighs[0], dists[0]
            # Format neighs information
            print neighs, dists, ke
            print np.any(neighs)
            print np.any(dists)
            if empty_neighs(neighs):
                continue
            nei_nfo.set(((neighs, dists), k))
            # Compute aggregation
            agg[i, :, k] = agg_f(features[nei_nfo], dists)

#        if any(neighs):
#            for k in range(sh[2]):
#                agg[i, :, k] = agg_f(features[(neighs, dists), k], dists)
#        else:
#            agg[i, :, :] = np.ones((sh[1], sh[2])) * _nullvalue

    ## 2. Prepare output
    agg = ExplicitFeatures(agg, indices=u_regs, characterizer=characterizer)
    ## TODO: names=[], nullvalue=None
    return agg
