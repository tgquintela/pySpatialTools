
"""
Avg descriptors
---------------
Module which groups the methods related with computing average-based spatial
descriptors.

"""

from descriptormodel import BaseDescriptorModel

## Specific functions
from ..aux_descriptormodels import avg_reducer, null_completer,\
    aggregator_summer, sum_addresult_function, general_featurenames,\
    null_out_formatter
## Characterizers
from ..aux_descriptormodels import characterizer_average,\
    characterizer_average_array, characterizer_average_listdict,\
    characterizer_average_listarray, characterizer_average_arrayarray


class AvgDescriptor(BaseDescriptorModel):
    """Model of spatial descriptor computing by averaging the type of the
    neighs represented in feat_arr.

    """
    name_desc = "Average descriptor"
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
        self.set_functions(type_infeatures)
        ## Check descriptormodel
        self._assert_correctness()

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute(self, pointfeats, point_pos):
        """Compulsary function to pass for the feture retriever. Compute
        average of features.

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
        return descriptors

#    def reducer(self, aggdescriptors_idxs, point_aggpos):
#        """Reducer gets the aggdescriptors of the neighbourhood regions
#        aggregated and collapse all of them to compute the descriptor
#        associated to a retrieved neighbourhood.
#        TODO: Global info for averaging
#        """
#        descriptors = avg_reducer(aggdescriptors_idxs, point_aggpos)
#        return descriptors
#
#    def aggdescriptor(self, pointfeats, point_pos):
#        "This function assigns descriptors to a aggregation unit."
#        descriptors = aggregator_summer(pointfeats, point_pos)
#        return descriptors

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, measure):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.

        TODO: count_vals and average

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
        self._out_formatter = null_out_formatter
        self._f_default_names = general_featurenames
#        self._defult_add2result = sum_addresult_function

    def set_functions(self, type_infeatures, type_outfeatures=None):
        """Set specific functions knowing a constant input and output desired.

        Parameters
        ----------
        type_infeatures: str, optional
            type of the input features.
        type_outfeatures: str, optional (default=None)
            type of the output features.

        """
        ## Preparing the clas for the known input
        if type_infeatures is None:
            self._core_characterizer = characterizer_average
        elif type_infeatures in ['array', 'ndarray']:
            self._core_characterizer = characterizer_average_array
        elif type_infeatures in ['list', 'listdict']:
            self._core_characterizer = characterizer_average_listdict
        elif type_infeatures == 'listarray':
            self._core_characterizer = characterizer_average_listarray
        elif type_infeatures == 'arrayarray':
            self._core_characterizer = characterizer_average_arrayarray

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################
