
"""
DescriptorModel
---------------
Module which contains the abstract class method for computing descriptor
models from puntual features.

"""

import numpy as np
from aux_descriptormodels.add2result_functions import sum_addresult_function
from aux_descriptormodels.featurenames_functions import array_featurenames
from aux_descriptormodels.out_formatters import null_out_formatter
## Specific functions
from pySpatialTools.FeatureManagement.Interpolation_utils import\
    create_weighted_function


class DescriptorModel:
    """Abstract class for desctiptor models. Contain the common functions
    that are required for each desctiptor instantiated object.
    """

    def complete_desc_i(self, i, neighs_info, desc_i, desc_neighs, vals_i):
        """Dummy completion for general abstract class."""
        desc = []
        for k in xrange(len(vals_i)):
            desc.append(self.relative_descriptors(i, neighs_info, desc_i[k],
                                                  desc_neighs[k], vals_i[k]))

#        descriptors = np.vstack(desc)
        return desc

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        "General default relative descriptors."
        return desc_neigh

    ####################### Compulsary general functions ######################
    ###########################################################################
    def _checker_descriptormodel(self):
        """Function to check if the desctiptormodel is well coded."""
        pass

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

    ###########################################################################
    ########################## Formatter functions ############################
    ###########################################################################


class GeneralDescriptor(DescriptorModel):
    """General descriptor model totally personalizable.
    """

    name_desc = "General descriptor"

    def __init__(self, interpolator, reducer, aggdescriptor, completer=None,
                 out_formatter=None, featurenames=None, add2result=None):
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
        self.compute_characs = interpolator
        self.reducer = reducer
        self.aggdescriptor = aggdescriptor
        if completer is not None:
            self.to_complete_measure = completer
        self._format_extra_functions(out_formatter, featurenames, add2result)
        self._checker_descriptormodel()

    def _format_extra_functions(self, out_formatter, featurenames, add2result):
        ## Initial function set
        if out_formatter is None:
            self._out_formatter = null_out_formatter
        else:
            self._out_formatter = out_formatter
        if featurenames is None:
            self._f_default_names = array_featurenames
        else:
            self._f_default_names = featurenames
        if add2result is None:
            self._defult_add2result = sum_addresult_function
        else:
            self._defult_add2result = add2result


class Interpolator(DescriptorModel):
    """
    TODO
    ----
    Matrix of weights between distances and own properties.
    (Weights not depending only on distances but also on ij type)
    Integrate special parameters in the functions with function-creators.

    implementation of 3 cases:
    - autoretrieve
    - non-autoretrieve without features_i
    - non-autoretrieve with features_i

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
        self.compute_characs = characterizer
        self.reducer = characterizer
        self.aggdescriptor = characterizer
        ## Initial function set
        self._out_formatter = null_out_formatter
        self._f_default_names = array_featurenames
        self._defult_add2result = sum_addresult_function
        self._checker_descriptormodel()
