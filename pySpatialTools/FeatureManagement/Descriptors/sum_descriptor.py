
"""
Sum descriptors
---------------
Module which groups the methods related with computing sum-based spatial
descriptors.

"""

from descriptormodel import BaseDescriptorModel

## Specific functions
from ..aux_descriptormodels import sum_reducer, null_completer,\
    null_out_formatter, array_featurenames, aggregator_summer

## Characterizers
from ..aux_descriptormodels import characterizer_summer,\
    characterizer_summer_array, characterizer_summer_listdict,\
    characterizer_summer_listarray, characterizer_summer_arrayarray


class SumDescriptor(BaseDescriptorModel):
    """Model of spatial descriptor computing by averaging the type of the
    neighs represented in feat_arr.

    Parameters
    ----------
    features: numpy.ndarray, shape (n, 2)
        the element-features of the system.
    sp_typemodel: str, object, ...
        the information of the type global output return.

    """
    name_desc = "Sum descriptor"
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
        """Compulsary function to pass for the feture retriever. It sums the
        features in the neighbourhood.

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
#        """
#        descriptors = sum_reducer(aggdescriptors_idxs, point_aggpos)
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

        Parameters
        ----------
        measure: np.ndarray or list dicts
            the measure computed by the whole spatial descriptor model.

        Returns
        -------
        measure: np.ndarray or list dicts
            the transformed measure computed by the whole spatial descriptor
            model.

        """
        measure = null_completer(measure, None)
        return measure

    ###########################################################################
    ########################## Auxiliary functions ############################
    ###########################################################################
    def _format_default_functions(self):
        """Format default mutable functions."""
        self._out_formatter = null_out_formatter
        self._core_characterizer = characterizer_summer
        self._f_default_names = array_featurenames
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
            self._core_characterizer = characterizer_summer
        elif type_infeatures in ['array', 'ndarray']:
            self._core_characterizer = characterizer_summer_array
        elif type_infeatures in ['list', 'listdict']:
            self._core_characterizer = characterizer_summer_listdict
        elif type_infeatures == 'listarray':
            self._core_characterizer = characterizer_summer_listarray
        elif type_infeatures == 'arrayarray':
            self._core_characterizer = characterizer_summer_arrayarray

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################
