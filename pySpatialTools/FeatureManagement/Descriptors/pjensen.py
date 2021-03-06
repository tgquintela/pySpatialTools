
"""
pjensen descriptors
-------------------
Module which groups the methods related with computing pjensen descriptors.


Parameters
----------

"""

import numpy as np
from collections import Counter
from descriptormodel import BaseDescriptorModel

## Specific functions
from ..aux_descriptormodels import\
    characterizer_1sh_counter, sum_reducer, counter_featurenames,\
    aggregator_1sh_counter, null_out_formatter,\
    count_out_formatter_general, count_out_formatter_dict2array


class PjensenDescriptor(BaseDescriptorModel):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in feat_arr.

    """
    name_desc = "Pjensen descriptor"
    _nullvalue = 0

    def __init__(self, features=None, type_infeatures=None,
                 type_outfeatures=None):
        """The inputs are the needed to compute model_dim.

        Parameters
        ----------
        features: np.ndarray (default=None)
            the features in a array_like mode.
        type_infeatures: str, optional (default=None)
            type of the input features.
        type_outfeatures: str, optional (default=None)
            type of the output features.

        """
        ## Global initialization
        self.default_initialization()
        ## Initial function set
        self.selfdriven = False
        self._format_default_functions()
        self.set_functions(type_infeatures, type_outfeatures)
        if features is not None:
            self.set_global_info(features)
        ## Check descriptormodel
        self._assert_correctness()

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute(self, pointfeats, point_pos):
        """Compulsary function to pass for the feture retriever.

        Parameters
        ----------
        pointfeats: list of arrays, np.ndarray or list of list of dicts
            the point features information. [iss][nei][feats]
        point_pos: list of arrays or np.ndarray.
            the element relative position of the neighbourhood.
            [iss][nei][rel_pos]

        Returns
        -------
        descriptors: list of arrays or np.ndarray or list of dicts
            the descriptor of the neighbourhood. [iss][feats]

        """
        descriptors = self._core_characterizer(pointfeats, point_pos)
        ## TODO: Transform dict to array and reverse
        #keys = [self.mapper[key] for key in counts.keys()]
        #descriptors[0, keys] = counts.values()
        return descriptors

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        """Completion of individual descriptor of neighbourhood by crossing the
        information of the individual descriptor of point to its neighbourhood
        descriptor.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        neighs_info:
            the neighbourhood information of each `i`.
        desc_i: np.ndarray or list of dict
            the descriptors associated to the elements `iss`.
        desc_neighs: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss`.
        vals_i: list or np.ndarray
            the storable index information.

        Returns
        -------
        desc_neigh: list
            the descriptors for each element.

        """
        descriptors = []
        for iss_i in range(len(desc_neigh)):
            descriptors.append(compute_loc_M_index(vals_i, desc_neigh[iss_i],
                               self.globals_))
        descriptors = np.array(descriptors)
        return descriptors
#
#    def reducer(self, aggdescriptors_idxs, point_aggpos):
#        """Reducer gets the aggdescriptors of the neighbourhood regions
#        aggregated and collapse all of them to compute the descriptor
#        associated to a retrieved neighbourhood.
#        """
#        descriptors = sum_reducer(aggdescriptors_idxs, point_aggpos)
#        return descriptors
#
#    def aggdescriptor(self, pointfeats, point_pos):
#        "This function assigns descriptors to a aggregation unit."
#        descriptors = aggregator_1sh_counter(pointfeats, point_pos)
#        return descriptors

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, measure):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.

        Parameters
        ----------
        measure: np.ndarray
            the measure computed by the whole spatial descriptor model.

        Returns
        -------
        measure: np.ndarray
            the transformed measure computed by the whole spatial descriptor
            model.

        """
        measure = normalization_jensen(measure, self.globals_)
        return measure

    def set_global_info(self, features):
        """Set the global stats info in order to get information to normalize
        the measure.

        Parameters
        ----------
        features: np.ndarray (default=None)
            the features in a array_like mode.

        """
        ##Globals
        if len(features.shape) >= 2:
            assert(np.prod(features.shape[1:]) == 1)
            features = features.ravel().astype(int)
        ## Compute the global information
        counts = dict(Counter(features))
        n_vals = len(np.unique(features))
        n = len(features)
        self.globals_ =\
            counts, n_vals, global_constants_jensen(n_vals, n, counts)

    def _format_default_functions(self):
        """Format default mutable functions."""
        self._out_formatter = count_out_formatter_general
        self._f_default_names = counter_featurenames
        self._core_characterizer = characterizer_1sh_counter
#        self._defult_add2result = sum_addresult_function

    def set_functions(self, type_infeatures, type_outfeatures):
        """Set specific functions knowing a constant input and output desired.

        Parameters
        ----------
        type_infeatures: str, optional (default=None)
            type of the input features.
        type_outfeatures: str, optional (default=None)
            type of the output features.

        """
        if type_outfeatures == 'dict':
            self._out_formatter = null_out_formatter
        else:
            self._out_formatter = count_out_formatter_dict2array


###############################################################################
####################### Asociated auxiliary functions #########################
###############################################################################
def compute_loc_M_index(idx, counts_i, globals_, sm_par=1e-10):
    """Computing the M index."""
    ## 0. Needed variables
    N_x, n_vals, C = globals_
    idx_val = N_x.keys()[idx]
    ## Compute the correlation contribution
    counts_i[idx] -= 1
    tot = counts_i.sum()
    if tot == 0:
        corr_loc_i = np.ones(n_vals)*sm_par
    elif counts_i[idx] == tot:
        corr_loc_i = np.zeros(n_vals)
        aux = float(tot)+N_x[idx_val]*sm_par
        corr_loc_i[idx] = (counts_i[idx]+sm_par)/aux
    else:
        aux = float(tot-counts_i[idx]+N_x[idx_val]*sm_par)
        corr_loc_i = (counts_i+sm_par)/aux
        aux = (float(tot)+N_x[idx_val]*sm_par)
        corr_loc_i[idx] = (counts_i[idx]+sm_par)/aux
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
    rep2idx = N_x.keys()
    for i in range(n_vals):
        for j in range(n_vals):
            i_val, j_val = rep2idx[i], rep2idx[j]
            if i == j:
                if N_x[i_val] <= 1:
                    C[i, j] = 0.
                else:
                    C[i, j] = (N_t-1)/float(N_x[i_val]*(N_x[i_val]-1))
            else:
                if N_x[i_val] == 0 or N_x[j_val] == 0:
                    C[i, j] = 0.
                else:
                    C[i, j] = (N_t-N_x[i_val])/float(N_x[i_val]*N_x[j_val])
    C[C < 0] = 0
    return C


def normalization_jensen(corr_loc, globals_):
    """Main function to compute the complete normalized measure of pjensen
    from the matrix of estimated counts.
    """
    ## 0. Needed variables
    _, n_vals, C = globals_
    ## 1. Computing the nets
    n_calc = corr_loc.shape[2]
    net = np.zeros((n_vals, n_vals, n_calc))
    for i in range(n_calc):
        idx_null = np.logical_or(C == 0, corr_loc[:, :, i] == 0)  # Probably incorrect
        net[:, :, i] = np.log10(np.multiply(C, corr_loc[:, :, i]))
        net[idx_null] = 0.
    return net
