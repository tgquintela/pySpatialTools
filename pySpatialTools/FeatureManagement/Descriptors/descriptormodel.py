
"""
DescriptorModel
---------------
Module which contains the abstract class method for computing descriptor
models from puntual features.

"""

from ..aux_descriptormodels.add2result_functions import sum_addresult_function
from ..aux_descriptormodels.featurenames_functions import array_featurenames,\
    list_featurenames
from ..aux_descriptormodels.out_formatters import null_out_formatter
## Specific functions
from pySpatialTools.FeatureManagement.Interpolation_utils import\
    create_weighted_function

#import numpy as np


class DescriptorModel:
    """Abstract class for desctiptor models. Contain the common functions
    that are required for each desctiptor instantiated object.
    """

    def default_initialization(self):
        self.selfdriven = True

    ####################### Factorized general functions ######################
    ###########################################################################
    def complete_desc_i(self, i, neighs_info, desc_i, desc_neighs, vals_i):
        """Dummy completion for general abstract class."""
#        ## TESTING CODE ########################
#        # Ensure correct inputs
#        if type(desc_neighs) == np.ndarray:
#            print desc_neighs.shape, vals_i, i
#            assert(len(desc_neighs.shape) == 3)
#        elif type(desc_neighs) == list:
#            assert(type(desc_neighs[0][0]) == dict)
#        assert(len(desc_i) == len(desc_neighs))
#        assert(len(vals_i) == len(desc_neighs))
#        ########################################
        desc = []
        for k in xrange(len(vals_i)):
            desc.append(self.relative_descriptors(i, neighs_info, desc_i[k],
                                                  desc_neighs[k], vals_i[k]))
        return desc

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        "General default relative descriptors."
        return desc_neigh

    ####################### Compulsary general functions ######################
    ###########################################################################
    def _assert_correctness(self):
        """Assert correct instantiation of the class."""
        assert('name_desc' in dir(self))
        assert('compute' in dir(self))

    ################# Dummy compulsary overwritable functions #################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return corr_loc

    def set_global_info(self, features):
        """Set the global stats info in order to get information to normalize
        the measure.
        """
        pass

    def set_functions(self, type_infeatures, type_outfeatures):
        """Dummy set for setting specific inputs and outputs."""
        ## Set outformatter as null
        self._out_formatter = null_out_formatter

    ###########################################################################
    ########################## Formatter functions ############################
    ###########################################################################


class DummyDescriptor(DescriptorModel):
    """Dummy descriptor model created for testing purposes."""

    name_desc = "Dummy descriptor"

    def __init__(self):
        self._out_formatter = null_out_formatter
        ## Check descriptormodel
        self._assert_correctness()

    def compute(self, pointfeats, point_pos):
        """From [iss][nei][feats] to [iss][feats]"""
        return [pfeats[0] for pfeats in pointfeats]

    def set_functions(self, type_infeatures, type_outfeatures):
        if type_infeatures == 'ndarray':
            self._f_default_names = array_featurenames
        else:
            self._f_default_names = list_featurenames


class GeneralDescriptor(DescriptorModel):
    """General descriptor model totally personalizable.
    """

    name_desc = "General descriptor"

    def __init__(self, characterizer, completer=None, out_formatter=None,
                 featurenames=None):
        """
        Parameters
        ----------
        weighting_avg: function
            the function created from the functions and parameters coded in
            the interpolation module.
        features: py.FeatureRetriever
            the pst data type which contains the whole information of element
            features and retrieving spatial features.
        """
        ## Specific class settings
        self.compute = characterizer
        if completer is not None:
            self.to_complete_measure = completer
        self._format_extra_functions(out_formatter, featurenames)
        self._assert_correctness()

    def _format_extra_functions(self, out_formatter, featurenames):
        ## Initial function set
        if out_formatter is None:
            self._out_formatter = null_out_formatter
        else:
            self._out_formatter = out_formatter
        if featurenames is None:
            self._f_default_names = array_featurenames
        else:
            self._f_default_names = featurenames


class Interpolator(DescriptorModel):
    """
    TODO
    ----
    Matrix of weights between distances and own properties.
    (Weights not depending only on distances but also on ij type)
    Integrate special parameters in the functions with function-creators.

    implementation of 4 cases:
    - autoretrieve
    - non-autoretrieve without features_i
    - non-autoretrieve with features_i
    - Fit locally more functions

    """

    name_desc = "Density assignation descriptor"

    def __init__(self, f_weight, pars_w, f_dens, pars_d):
        """
        Parameters
        ----------
        weighting_avg: function
            the function created from the functions and parameters coded in
            the interpolation module.
        features: py.FeatureRetriever
            the pst data type which contains the whole information of element
            features and retrieving spatial features.
        """
        characterizer = create_weighted_function(f_weight, pars_w,
                                                 f_dens, pars_d)
        ## Specific class settings
        self.compute = characterizer
        ## Initial function set
        self._out_formatter = null_out_formatter
        self._f_default_names = array_featurenames
        self._defult_add2result = sum_addresult_function
        self._assert_correctness()