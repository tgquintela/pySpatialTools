
"""
Sparse counter
--------------
Counting values in a sparse way.


"""

from count_descriptor import Countdescriptor

## Specific functions
from ..aux_descriptormodels import append_addresult_function,\
    count_out_formatter_general, sparse_dict_completer, counter_featurenames


class SparseCounter(Countdescriptor):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in features in a sparse fashion.
    """

    name_desc = "Sparse counting descriptor"
    _nullvalue = 0

    def __init__(self):
        """The inputs are the needed to compute model_dim."""
        ## Initial function set
        self._out_formatter = count_out_formatter_general
        self._f_default_names = counter_featurenames
        self._defult_add2result = append_addresult_function
        ## Check descriptormodel
        self._checker_descriptormodel()

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    # Herency from Countdescriptor

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        corr_loc = sparse_dict_completer(corr_loc)
        return corr_loc

    ###########################################################################
    ########################## Auxiliary functions ############################
    ###########################################################################

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################
