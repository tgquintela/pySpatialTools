
"""
Sparse counter
--------------
Counting values in a sparse way.


"""

from count_descriptor import CountDescriptor

## Specific functions
from ..aux_descriptormodels import append_addresult_function,\
    count_out_formatter_general, sparse_dict_completer, counter_featurenames


class SparseCounter(CountDescriptor):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in features in a sparse fashion.
    """

    name_desc = "Sparse counting descriptor"
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
    # Herency from Countdescriptor

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, measure):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.

        Parameters
        ----------
        measure: list [ks][vals_i]{feats}
            the measure computed by the whole spatial descriptor model.

        Returns
        -------
        measure:  list of scipy.sparse
            the transformed measure computed by the whole spatial descriptor
            model.

        """
        measure = sparse_dict_completer(measure)
        return measure

    ###########################################################################
    ########################## Auxiliary functions ############################
    ###########################################################################
    # Herency from Countdescriptor

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################
