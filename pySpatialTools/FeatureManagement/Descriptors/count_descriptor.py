
"""
Count descriptors
-----------------
Module which groups the methods related with computing histogram-based spatial
descriptors.


"""

from ..descriptormodel import DescriptorModel

## Specific functions
from ..aux_descriptormodels import\
    characterizer_1sh_counter, sum_reducer, null_completer,\
    aggregator_1sh_counter, sum_addresult_function, counter_featurenames,\
    count_out_formatter


class Countdescriptor(DescriptorModel):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in feat_arr.

    Parameters
    ----------

    """
    name_desc = "Counting descriptor"
    _nullvalue = 0

    def __init__(self, sp_typemodel='matrix'):
        """The inputs are the needed to compute model_dim."""
        ## Initial function set
        self._out_formatter = count_out_formatter
        self._f_default_names = counter_featurenames
        self._defult_add2result = sum_addresult_function
        ## Check descriptormodel
        self._checker_descriptormodel()

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
        corr_loc = null_completer(corr_loc)
        return corr_loc

    ###########################################################################
    ########################## Auxiliary functions ############################
    ###########################################################################

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################
