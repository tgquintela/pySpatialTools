
"""
Sum descriptors
---------------
Module which groups the methods related with computing sum-based spatial
descriptors.

"""

from pySpatialTools.Feature_engineering.descriptormodel import DescriptorModel

## Specific functions
from pySpatialTools.Feature_engineering.aux_descriptormodels import\
    characterizer_summer, sum_reducer, null_completer, aggregator_summer,\
    sum_addresult_function, array_featurenames


class AvgDescriptor(DescriptorModel):
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
    _n = 0
    _nullvalue = 0

    def __init__(self, features, sp_typemodel='matrix'):
        "The inputs are the needed to compute model_dim."
        ## Initial function set
        self._f_default_names = array_featurenames
        self._defult_add2result = sum_addresult_function
        ## Format features
        self._format_features(features)
        ## Type of built result
        self._format_map_vals_i(sp_typemodel)
        ## Format function to external interaction and building results
        self._format_result_building()

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_characs(self, pointfeats, point_pos):
        "Compulsary function to pass for the feture retriever."
        descriptors = characterizer_summer(pointfeats, point_pos)
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

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################
