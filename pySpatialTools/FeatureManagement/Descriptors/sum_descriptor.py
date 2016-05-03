
"""
Sum descriptors
---------------
Module which groups the methods related with computing sum-based spatial
descriptors.

"""

from ..descriptormodel import DescriptorModel

## Specific functions
from ..aux_descriptormodels import sum_reducer, null_completer,\
    null_out_formatter, array_featurenames, aggregator_summer

## Characterizers
from ..aux_descriptormodels import characterizer_summer,\
    characterizer_summer_array, characterizer_summer_listdict,\
    characterizer_summer_listarray, characterizer_summer_arrayarray


class SumDescriptor(DescriptorModel):
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
        """The inputs are the needed to compute model_dim."""
        ## Initial function set
        self._format_default_functions()
        self.set_functions(type_infeatures)
        ## Check descriptormodel
        self._checker_descriptormodel()

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_characs(self, pointfeats, point_pos):
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
        descriptors = aggregator_summer(pointfeats, point_pos)
        return descriptors

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        corr_loc = null_completer(corr_loc, None)
        return corr_loc

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
