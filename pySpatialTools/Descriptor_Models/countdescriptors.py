

"""
Count descriptors
-------------------
Module which groups all the functions related with the computation of the
spatial correlation using counting descriptors.

TODO
----
- Support for more than 1 dimensional type_var.
"""

import numpy as np
from descriptor_models import DescriptorModel
from pySpatialTools.Preprocess.aggregation_utils import \
    compute_aggregate_counts
from aux_functions import compute_global_counts


########### Class for computing index of the model selected
##################################################################
class Countdescriptor(DescriptorModel):
    """
    Model of spatial descriptor computing by counting the type of the neighs.

    TODO
    ----

    """
    name_desc = "Counting descriptors"

    def __init__(self, df, typevars):
        "The inputs are the needed to compute model_dim."
        self.typevars = typevars
        self.counts, self.counts_info = compute_globalstats(df, typevars)
        self.n_vals = self.counts_info.shape[0]
        self.model_dim = self.compute_model_dim()
        self.globalnorm = 1

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def to_complete_measure(self, corr_loc, N_t):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return corr_loc

    def compute_descriptors(self, characs, val_i):
        """Compute descriptors, main funtion of the son class. It returns the
        descriptors of the element evaluated by computing from its trivial
        descriptors and its own type value (val_i).
        """
        characs[val_i] -= 1
        return characs

    ###########################################################################
    ####################### Compulsary functions agg ##########################
    ###########################################################################
    def compute_aggcharacterizers(self, df, agg_var, type_vars, reindices):
        """Compute aggregate descriptors. It returns the aggregate
        descriptors for the whole data.
        """
        desc, _ = compute_aggregate_counts(df, agg_var, type_vars, reindices)
        desc = desc[desc.keys()[0]]
        return desc

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
        counts_i = np.ravel(counts_i)
        return counts_i

    ###########################################################################
    ######################## Auxiliar functions corr ##########################
    ###########################################################################


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


def count_in_neighborhood(vals, n_vals):
    "Counting neighbours in the neighbourhood."
    counts_i = [np.count_nonzero(np.equal(vals, v)) for v in range(n_vals)]
    counts_i = np.array(counts_i)
    return counts_i


def aggregate_precomp_descriptors(vals, n_vals):
    "Aggregate local sums (bag of words)."
    counts_i = np.sum(vals, axis=0)
    return counts_i


def compute_globalstats(df, typevars):
    feat_vars = typevars['feat_vars']
    counts, counts_info = compute_global_counts(df, feat_vars)
    counts = counts[counts.keys()[0]]
    counts_info = counts_info[counts_info.keys()[0]]
    counts, counts_info = np.array(counts), np.array(counts_info)
    return counts, counts_info
