
"""
Count descriptors
-----------------
Module which groups the methods related with computing histogram-based spatial
descriptors.

"""

import numpy as np
from descriptormodel import BaseDescriptorModel

## Specific functions
from ..aux_descriptormodels import\
    characterizer_1sh_counter, sum_reducer, null_completer,\
    aggregator_1sh_counter, counter_featurenames,\
    count_out_formatter_general, null_out_formatter,\
    count_out_formatter_dict2array


class CountDescriptor(BaseDescriptorModel):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in feat_arr.

    """
    name_desc = "Counting descriptor"
    _nullvalue = 0

    def __init__(self, type_infeatures=None, type_outfeatures=None):
        """The inputs are the needed to compute model_dim.

        Parameters
        ----------
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
        ## Check descriptormodel
        self._assert_correctness()

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute(self, pointfeats, point_pos):
        """Compulsary function to pass for the feture retriever. Counts for
        each of the possible types of elements in the neighbourhood.

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
        measure = null_completer(measure)
        return measure

    ###########################################################################
    ########################## Auxiliary functions ############################
    ###########################################################################
    def _format_default_functions(self):
        """Format default mutable functions."""
        self._out_formatter = count_out_formatter_general
        self._core_characterizer = characterizer_1sh_counter
        self._f_default_names = counter_featurenames
#        self._defult_add2result = sum_addresult_function

    def set_functions(self, type_infeatures, type_outfeatures):
        """Set specific functions knowing a constant input and output desired.

        Parameters
        ----------
        type_infeatures: str, optional
            type of the input features.
        type_outfeatures: str, optional
            type of the output features.

        """
        if type_outfeatures == 'dict':
            self._out_formatter = null_out_formatter
        else:
            self._out_formatter = count_out_formatter_dict2array

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################


class CounterNNDesc(BaseDescriptorModel):
    """Descriptor based on count all the neighs that it receives."""
    name_desc = "Counter NN descriptor"
    _nullvalue = 0

    def __init__(self):
        """The inputs are the needed to compute model_dim."""
        ## Global initialization
        self.default_initialization()
        ## Initial function set
        self.selfdriven = False
        self._format_default_functions()
        ## Check descriptormodel
        self._assert_correctness()

    def compute(self, pointfeats, point_pos):
        """Compute descriptors by counting neighs in the neighbourhood.

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
        n_iss = len(pointfeats)
        descriptors = np.zeros((n_iss, 1))
        for i in range(n_iss):
            descriptors[i] = len(pointfeats[i])
        return descriptors

    def _format_default_functions(self):
        """Format default mutable functions."""
        self._out_formatter = null_out_formatter
        self._f_default_names = lambda x: [0]
