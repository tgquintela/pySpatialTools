
"""
pjensen descriptors
-------------------
Module which groups the methods related with computing pjensen descriptors.


Parameters
----------

"""

import numpy as np
from collections import Counter
from ..descriptormodel import DescriptorModel

## Specific functions
from ..aux_descriptormodels import\
    characterizer_1sh_counter, sum_reducer, counter_featurenames,\
    aggregator_1sh_counter, sum_addresult_function, count_out_formatter


class Countdescriptor(DescriptorModel):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in feat_arr.

    Parameters
    ----------
    features: numpy.ndarray, shape (n, 2)
        the element-features of the system.
    sp_typemodel: str, object, ...
        the information of the type global output return.

    """
    name_desc = "Pjensen descriptor"
    _n = 0
    _nullvalue = 0

    def __init__(self, features, sp_typemodel='matrix'):
        "The inputs are the needed to compute model_dim."
        ## Initial function set
        self._out_formatter = count_out_formatter
        self._f_default_names = counter_featurenames
        self._defult_add2result = sum_addresult_function
        ## Format features
        self._format_features(features)
        ## Type of built result
        self._format_map_vals_i(sp_typemodel)
        ## Format function to external interaction and building results
        self._format_result_building()
        ##Globals
        counts = dict(Counter(self.features.features[0][:, 0]))
        n_vals = np.unique(self.features.features[0][:, 0])
        self._n = len(self.features[0])
        self.globals_ =\
            counts, n_vals, global_constants_jensen(n_vals, self._n, counts)

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_characs(self, pointfeats, point_pos):
        "Compulsary function to pass for the feture retriever."
        descriptors = characterizer_1sh_counter(pointfeats, point_pos)
        ## TODO: Transform dict to array and reverse
        #keys = [self.mapper[key] for key in counts.keys()]
        #descriptors[0, keys] = counts.values()
        return descriptors

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        """Completion of individual descriptor of neighbourhood by crossing the
        information of the individual descriptor of point to its neighbourhood
        descriptor.
        """
        descriptors = compute_loc_M_index(desc_i, desc_neigh, self.globals_)
        return descriptors

    def reducer(self, aggdescriptors_idxs, point_aggpos):
        """Reducer gets the aggdescriptors of the neighbourhood regions
        aggregated and collapse all of them to compute the descriptor
        associated to a retrieved neighbourhood.
        """
        descriptors = sum_reducer(aggdescriptors_idxs, point_aggpos)
        return descriptors

    def aggdescriptor(self, pointfeats, point_pos):
        "This function assigns descriptors to a aggregation unit."
        descriptors = aggregator_1sh_counter(pointfeats, point_pos)
        return descriptors

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        corr_loc = normalization_jensen(corr_loc, self.globals_)
        return corr_loc


###############################################################################
####################### Asociated auxiliary functions #########################
###############################################################################
def compute_loc_M_index(idx, counts_i, globals_, sm_par=1e-10):
    "Computing the M index."
    ## 0. Needed variables
    _, N_x, n_vals, C = globals_
    ## Compute the correlation contribution
    counts_i[idx] -= 1
    tot = counts_i.sum()
    if tot == 0:
        corr_loc_i = np.ones(n_vals)*sm_par
    elif counts_i[idx] == tot:
        corr_loc_i = np.zeros(n_vals)
        corr_loc_i[idx] = (counts_i[idx]+sm_par)/(float(tot)+N_x[idx]*sm_par)
    else:
        corr_loc_i = (counts_i+sm_par)/float(tot-counts_i[idx]+N_x[idx]*sm_par)
        corr_loc_i[idx] = (counts_i[idx]+sm_par)/(float(tot)+N_x[idx]*sm_par)
    # Avoid nan values
    corr_loc_i[np.isnan(corr_loc_i)] = sm_par
    corr_loc_i[corr_loc_i < 0] = sm_par
    return corr_loc_i


def global_constants_jensen(n_vals, N_t, N_x):
    """Auxiliary function to compute the global constants of the the M index
    for spatial correlation. This constants represent the global density stats
    which are used as the null model to compare with the local stats.
    """
    ## Building the normalizing constants
    C = np.zeros((n_vals, n_vals))
    for i in range(n_vals):
        for j in range(n_vals):
            if i == j:
                if N_x[i] <= 1:
                    C[i, j] = 0.
                else:
                    C[i, j] = (N_t-1)/float(N_x[i]*(N_x[i]-1))
            else:
                if N_x[i] == 0 or N_x[j] == 0:
                    C[i, j] = 0.
                else:
                    C[i, j] = (N_t-N_x[i])/float(N_x[i]*N_x[j])
    C[C < 0] = 0
    return C


def normalization_jensen(corr_loc, globals_):
    """Main function to compute the complete normalized measure of pjensen
    from the matrix of estimated counts.
    """
    ## 0. Needed variables
    N_t, _, n_vals, C = globals_
    ## 1. Computing the nets
    n_calc = corr_loc.shape[2]
    net = np.zeros((n_vals, n_vals, n_calc))
    for i in range(n_calc):
        idx_null = np.logical_or(C == 0, corr_loc[:, :, i] == 0)  # Probably incorrect
        net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
        net[idx_null] = 0.
    return net


#def compute_globalstats(df, typevars):
#    feat_vars = typevars['feat_vars']
#    counts, counts_info = compute_global_counts(df, feat_vars)
#    counts = counts[counts.keys()[0]]
#    counts_info = counts_info[counts_info.keys()[0]]
#    counts, counts_info = np.array(counts), np.array(counts_info)
#    return counts, counts_info
