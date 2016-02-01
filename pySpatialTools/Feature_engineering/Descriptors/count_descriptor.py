
"""
Count descriptors
-----------------
Module which groups the methods related with computing histogram-based spatial
descriptors.


Parameters
----------
out

"""

from pySpatialTools.Feature_engineering.descriptormodel import DescriptorModel

## Specific functions
from pySpatialTools.Feature_engineering.aux_descriptormodels import\
    characterizer_1sh_counter, sum_reducer, null_completer,\
    aggregator_1sh_counter, sum_addresult_function, counter_featurenames


class Countdescriptor(DescriptorModel):
    """Model of spatial descriptor computing by counting the type of the
    neighs represented in feat_arr.

    Parameters
    ----------
    features: numpy.ndarray, shape (n, 2)
        the element-features of the system.
    sp_typemodel: str, object, ...
        the information of the type global output return.
    ####################################################### to featureretreiver
    mapper: dict
        map the type of feature to a number that could act as an index.

    """
    name_desc = "Counting descriptor"
    _n = 0
    _nullvalue = 0

    def __init__(self, features, sp_typemodel='matrix'):
        "The inputs are the needed to compute model_dim."
        ## Initial function set
        self._f_default_names = counter_featurenames
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
        corr_loc = null_completer(corr_loc, None)
        return corr_loc

    ###########################################################################
    ########################## Auxiliary functions ############################
    ###########################################################################

    ###########################################################################
    ######################### Compulsary formatters ###########################
    ###########################################################################


#    def compute_aggcharacs_i(self, neighs_i, dists_i):
#        """Compute aggregated characters for the region i from neighs_i points
#        and relative position of neighbourhood points dists_i.
#
#        Parameters
#        ----------
#        neighs_i: numpy.ndarray
#            the points which conforms the neighbourhood.
#        dists_i: numpy.ndarray
#            the relative position respect the original region.
#
#        Returns
#        -------
#        aggcharacs_i: numpy.ndarray
#            the information aggregated information features.
#
#        """
#        counts = Counter(self.features[list(neighs_i), :].ravel())
#        aggcharacs_i = self.initialization_desc()
#        keys = [self.mapper[key] for key in counts.keys()]
#        aggcharacs_i[0, keys] = counts.values()
#        return aggcharacs_i
