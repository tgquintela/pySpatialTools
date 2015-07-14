

"""
Module which groups all the functions related with the computation of the
spatial correlation using Jensen model.

TODO
----
- Support for more than 1 dimensional type_var.
"""

import numpy as np
from Mscthesis.Models import Model


########### Class for computing index of the model selected
##################################################################
class Pjensen(Model):
    """
    Model of spatial correlation inference. This model is the application of
    the spatial correlation used by P. Jensen [1]

    References
    ----------
    .. [1]

    """
    #bool_agg
    #bool_matrix
    #bool_r_agg
    #var_types
    def __init__():
        if agg_file_info is not None:
            self.agg_filepath = agg_file_info['filepath']
            self.locs_var_agg = agg_file_info['locs_vars']
            self.types_vars_agg = agg_file_info['type_vars']
            self.bool_agg = True
        

    def to_complete_measure(self, corr_loc, n_vals, N_t, N_x):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        n_vals = n_vals[0]
        C = global_constants_jensen(n_vals, N_t, N_x)
        # Computing the nets
        n_calc = corr_loc.shape[2]
        net = np.zeros((n_vals, n_vals, n_calc))
        for i in range(n_calc):
            idx_null = np.logical_or(C == 0, corr_loc[:, :, i] == 0)  # Probably incorrect
            net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
            net[idx_null] = 0.
        return net

    def compute_descriptors(self, vals, val_i, n_vals, C=None, idx_null=None,
                            N_x=None):
        """Compute descriptors, main funtion of the son class. It returns the
        descriptors of the element evaluated by computing from its trivial
        descriptors and its own type value (val_i).
        """
        ## 0. Needed variables transformations
        n_vals = n_vals[0]

        ## 1. Computation of the descriptors
        counts_i = compute_raw_descriptors(vals, n_vals, self.bool_agg)
        corr_loc_i = compute_local_descriptors(counts_i, val_i, n_vals, N_x)
        # Normalization
        if C is not None and idx_null is not None:
            descriptors = np.log10(np.multiply(C, corr_loc_i))
            descriptors[idx_null] = 0.
        else:
            descriptors = corr_loc_i
        return descriptors

    ###########################################################################
    ######################## Auxiliar functions corr ##########################
    ###########################################################################
    def compute_model_dim(self, n_vals, N_x):
        """Auxiliar function for computing the dimensions required for the
        result of the correlation. It is dependant with the model we select.
        """
        n_vals0, n_vals1 = n_vals[0], n_vals[0]
        return n_vals0, n_vals1

    def compute_global_info_descriptor(self, n_vals, N_t, N_x):
        """Function which groups in a dict all the needed global information to
        compute the desired measure. This information will be used by the
        specific model function compute_descriptors.
        """

        ## 0. Needed variables transformations
        n_vals = n_vals[0]
        N_x = N_x[N_x.keys()[0]]

        ## 1. Compute variables for compute_descriptors function
        if self.bool_matrix:
            C = global_constants_jensen(n_vals, N_t, N_x)
            idx_null = C == 0
            out = {'C': C, 'idx_null': idx_null, 'N_x': N_x}
        else:
            out = {'N_x': N_x}
        return out

    def get_characterizers(self, i, k, neighs, type_arr, reindices,
                           agg_desc=None):
        """Retrieve local characterizers for i element and k permutation. It
        returns the column index in the output matrix correlation (val_i) and
        trivial descriptors of the neighbourhood (vals). This values are used
        for the specific model function compute_descriptors.
        """
        if not self.bool_r_agg:  # self.bool_agg:
            val_i = type_arr[reindices[i, k], 0]
            neighs_k = reindices[neighs, k]
            vals = type_arr[neighs_k, :]
        else:
            var = self.var_types['type_vars'][0]
            val_i = type_arr[reindices[i, k], 0]
            vals = agg_desc[var][neighs, :, k]
        return val_i, vals

    ###########################################################################
    ############################ Quality measure ##############################
    ###########################################################################
    def compute_quality(self, corr_loc, count_matrix, type_arr, val_type=None):
        "Computation of the quality measure associated to the model."
        Q = compute_quality_measure(corr_loc, count_matrix, type_arr, val_type)
        return Q


def compute_raw_descriptors(vals, n_vals, bool_agg):
    """Computation of counts for the implemented methods nowadays.
    It is used to compute local raw descriptors or for aggregate the ones
    precomputed.
    """
    if bool_agg:
        counts_i = aggregate_precomp_descriptors(vals, n_vals)
    else:
        counts_i = count_in_neighborhood(vals, n_vals)
    return counts_i


def compute_local_descriptors(counts_i, idx, n_vals, N_x):
    """Compute the descriptor associated to this model.
    """
    ## Compute the correlation contribution
    corr_loc_i = compute_loc_M_index(counts_i, idx, n_vals, N_x)
    return corr_loc_i


def count_in_neighborhood(vals, n_vals):
    "Counting neighbours in the neighbourhood."
    counts_i = [np.count_nonzero(np.equal(vals, v)) for v in range(n_vals)]
    counts_i = np.array(counts_i)
    return counts_i


def aggregate_precomp_descriptors(vals, n_vals):
    "Aggregate local sums (bag of words)."
    counts_i = np.sum(vals, axis=0)
    return counts_i


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


def global_constants_jensen(n_vals, N_t, N_x):
    """Auxiliary function to compute the global constants of the the M index
    for spatial correlation. This constants represent the global density stats
    which are used as the null model to compare with the local stats.
    """
    ## Building the normalizing constants
    N_x = N_x[N_x.keys()[0]]
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


def compute_quality_measure(corr_loc, count_matrix, type_arr, val_type=None):
    "Main function to compute the quality measure of pjensen."
    ## Compute needed variables
    type_vals = np.unique(type_arr)
    n, n_vals = count_matrix.shape
    ## Loop over each type
    averages = np.zeros((n_vals, n_vals))
    for val_j in type_vals:
        averages[val_j, :] = np.mean(count_matrix[type_arr == val_j, :],
                                     axis=0)
    ## Loop for each
    Q = np.zeros(n)
    for i in range(n):
        if val_type is not None:
            val_j = val_type
        else:
            val_j = type_arr[i]
        avg = averages[val_j, :]
        Q[i] = np.sum(corr_loc[val_j, :] * (count_matrix[i, :] - avg))
    return Q
