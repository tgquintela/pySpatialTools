
"""
Descriptors
-----------
"""

import numpy as np
from collections import Counter
from ..descriptormodel import DescriptorModel


########### Class for computing index of the model selected
##################################################################
class Pjensen(DescriptorModel):
    """Model of spatial correlation inference. This model is the application of
    the spatial correlation used by P. Jensen [1]

    References
    ----------
    .. [1]


    TODO
    ----
    - All for numpy.ndarray

    """
    name_desc = "PJensen descriptors"

    nvals1_dim = 0
    counts = {}
    globalnorm = 0
    aggbool = False

    def __init__(self, feat_arr, aggbool=False):
        "The inputs are which are needed to compute global properties."
        self.globalstats, self.nvals1_dim, self.globalnorm =\
            compute_globalstats(feat_arr)

    def compute_descriptors(self, val_i, characs, idx_null=None):
        """Compute descriptors, main funtion of the son class. It returns the
        descriptors of the element evaluated by computing from its trivial
        descriptors and its own type value (val_i).
        """
        ## 0. Needed variables transformations
        n_vals = self.nvals1_dim
        N_x = self.globalstats
        C = self.globalnorm

        ## 1. Computation of the descriptors
        corr_loc_i = compute_local_descriptors(characs, val_i, n_vals, N_x)
        idx_null = np.logical_or(C[val_i, :] == 0, corr_loc_i == 0)  # Probably incorrect

        # Normalization
        descriptors = np.log10(np.multiply(C[val_i, :], corr_loc_i))
        descriptors[idx_null] = 0.

        return descriptors

    def to_complete_measure(self, corr_loc, N_t):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return corr_loc

    def to_add_descriptors(self, pre, res):
        return pre + res


def compute_globalstats(feat_arr):
    counts = dict(Counter(feat_arr))
    nvals1_dim = len(self.counts)
    globalnorm = global_constants_jensen(nvals1_dim, feat_arr.shape[0], counts)
    return nvals1_dim, counts, globalnorm



'''
Computation of characterizers
'''


def compute_local_descriptors(counts_i, idx, n_vals, N_x):
    """Compute the descriptor associated to this model.
    """
    ## Compute the correlation contribution
    corr_loc_i = compute_loc_M_index(counts_i, idx, n_vals, N_x)
    return corr_loc_i


def compute_loc_M_index(counts_i, idx, n_vals, N_x, sm_par=1e-10):
    "Computing the M index."
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
###############################################################################

'''
Computation of counts_i (or getting precomputed)
'''

def count_in_neighborhood(vals, n_vals):
    "Counting neighbours in the neighbourhood."
    counts_i = [np.count_nonzero(np.equal(vals, v)) for v in range(n_vals)]
    counts_i = np.array(counts_i)
    return counts_i



def aggregate_precomp_descriptors(vals, n_vals):
    "Aggregate local sums (bag of words)."
    counts_i = np.sum(vals, axis=0)
    return counts_i
###############################################################################


'''
Global computations (setting class or normalization)
'''
def compute_globalstats(df, typevars):
    feat_vars = typevars['feat_vars']
    counts, counts_info = compute_global_counts(df, feat_vars)
    counts = counts[counts.keys()[0]]
    counts_info = counts_info[counts_info.keys()[0]]
    counts, counts_info = np.array(counts), np.array(counts_info)
    return counts, counts_info


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


def normalization_jensen(corr_loc, N_t, n_vals, C):
    """Main function to compute the complete normalized measure of pjensen
    from the matrix of estimated counts.
    """
    ## 0. Needed variables
    ## 1. Computing the nets
    n_calc = corr_loc.shape[2]
    net = np.zeros((n_vals, n_vals, n_calc))
    for i in range(n_calc):
        idx_null = np.logical_or(C == 0, corr_loc[:, :, i] == 0)  # Probably incorrect
        net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
        net[idx_null] = 0.
    return net
