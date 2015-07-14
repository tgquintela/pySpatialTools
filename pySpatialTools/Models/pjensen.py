

"""
Module which groups all the functions related with the computation of the
spatial correlation using Jensen model.

TODO
----
- Support for more than 1 dimensional type_var.
"""

import numpy as np
from Mscthesis.Models import DescriptorModel
from Mscthesis.Preprocess.comp_complementary_data import \
    compute_aggregate_counts
from aux_functions import compute_global_counts


########### Class for computing index of the model selected
##################################################################
class Pjensen(DescriptorModel):
    """
    Model of spatial correlation inference. This model is the application of
    the spatial correlation used by P. Jensen [1]

    References
    ----------
    .. [1]

    TODO
    ----
    The model has to retrieve the val_i and the features (model_dim)

    """

    def __init__(self, df, typevars):
        "The inputs are the needed to compute model_dim."
        self.typevars = typevars
        self.counts, self.counts_info = compute_globalstats(df, typevars)
        self.n_vals = self.counts_info.shape[0]
        self.model_dim = self.compute_model_dim()
        self.globalnorm = self.compute_global_info_descriptor(df.shape[0])

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def to_complete_measure(self, corr_loc, N_t):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        ## 0. Needed variables
        n_vals = self.counts_info.shape[0]
        C = self.globalnorm
        ## 1. Computing the nets
        n_calc = corr_loc.shape[2]
        net = np.zeros((n_vals, n_vals, n_calc))
        for i in range(n_calc):
            idx_null = np.logical_or(C == 0, corr_loc[:, :, i] == 0)  # Probably incorrect
            net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
            net[idx_null] = 0.
        return net

    def compute_descriptors(self, characs, val_i, idx_null=None):
        """Compute descriptors, main funtion of the son class. It returns the
        descriptors of the element evaluated by computing from its trivial
        descriptors and its own type value (val_i).
        """
        ## 0. Needed variables transformations
        n_vals = self.counts_info.shape[0]
        N_x = self.counts
        C = self.globalnorm

        ## 1. Computation of the descriptors
        #counts_i = compute_raw_descriptors(vals, n_vals, self.bool_agg)
        corr_loc_i = compute_local_descriptors(characs, val_i, n_vals, N_x)
        idx_null = np.logical_or(C == 0, corr_loc_i == 0)  # Probably incorrect

        # Normalization
        descriptors = np.log10(np.multiply(C, corr_loc_i))
        descriptors[idx_null] = 0.

        return descriptors

    def compute_aggcharacterizers(self, df, agg_var, type_vars, reindices):
        """Compute aggregate descriptors. It returns the aggregate
        descriptors for the whole data.
        """
        desc, _ = compute_aggregate_counts(df, agg_var, type_vars, reindices)
        des = des[des.keys()[0]]
        return desc

    ###########################################################################
    ######################## Auxiliar functions corr ##########################
    ###########################################################################
    def compute_global_info_descriptor(self, N_t):
        """Function which groups in a dict all the needed global information to
        compute the desired measure. This information will be used by the
        specific model function compute_descriptors.
        """
        ## 0. Needed variables transformations
        n_vals = self.counts_info.shape[0]
        N_x = self.counts
        ## 1. Computation of the normalization constant
        C = global_constants_jensen(n_vals, N_t, N_x)
        return C

    ###########################################################################
    ######################## Auxiliar class functions #########################
    ###########################################################################
    def compute_model_dim(self):
        """Auxiliar function for computing the dimensions required for the
        result of the correlation. It is dependant with the model we select.
        """
        n_vals0, n_vals1 = self.counts_info.shape[0], self.counts_info.shape[0]
        return n_vals0, n_vals1

    def compute_value_i(self, i, k, feat_arr, reindices):
        "Compute the val of a specific point."
        val_i = feat_arr[reindices[i, k], :].astype(int)
        return val_i

    def compute_vals_nei(self, aggfeatures, feat_arr, neighs, reindices, k,
                         type_n):
        "Function which retrieve the vals from neighs and feature arrays."
        if type_n == 'aggregate':
            vals = aggfeatures[neighs, :, k].astype(int)
        else:
            vals = feat_arr[reindices[neighs, k]].astype(int)
        return vals

    def integrate_vals(self, vals, type_n):
        """Integrate the vals once they are retrieved the values from features
        and neighbours.
        """
        if type_n == 'aggregate':
            counts_i = np.sum(vals, axis=0)
        else:
            n_vals = self.counts_info.shape[0]
            counts_i = count_in_neighborhood(vals, n_vals)
        return counts_i

    ###########################################################################
    ############################ Quality measure ##############################
    ###########################################################################
    def compute_quality(self, corr_loc, count_matrix, feat_arr, val_type=None):
        "Computation of the quality measure associated to the model."
        Q = compute_quality_measure(corr_loc, count_matrix, feat_arr, val_type)
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
