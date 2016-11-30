
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


class BaseDescriptorModel:
    """Abstract class for desctiptor models. Contain the common functions
    that are required for each desctiptor instantiated object.
    """

    def default_initialization(self):
        self.selfdriven = True

    ####################### Factorized general functions ######################
    ###########################################################################
    def complete_desc_i(self, i, neighs_info, desc_i, desc_neighs, vals_i):
        """Dummy completion for general abstract class.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        neighs_info:
            the neighbourhood information of each `i`.
        desc_i: np.ndarray or list of dict
            the descriptors associated to the elements `iss` for each
            perturbation `k`.
        desc_neighs: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss` for each perturbation `k`.
        vals_i: list or np.ndarray
            the storable index information for each perturbation `k`.

        Returns
        -------
        desc: list
            the descriptors for each perturbation and element.

        """
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
        """General default relative descriptors.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        neighs_info:
            the neighbourhood information of each `i`.
        desc_i: np.ndarray or list of dict
            the descriptors associated to the elements `iss`.
        desc_neighs: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss`.
        vals_i: list or np.ndarray
            the storable index information.

        Returns
        -------
        desc_neigh: list
            the descriptors for each element.

        """
        return desc_neigh

    ####################### Compulsary general functions ######################
    ###########################################################################
    def _assert_correctness(self):
        """Assert correct instantiation of the class."""
        assert('name_desc' in dir(self))
        assert('compute' in dir(self))

    ################# Dummy compulsary overwritable functions #################
    def to_complete_measure(self, measure):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.

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
        return measure

    def set_global_info(self, features):
        """Set the global stats info in order to get information to normalize
        the measure.

        Parameters
        ----------
        features: np.ndarray or list (default=None)
            the features in a array_like mode or list dict mode.

        """
        pass

    def set_functions(self, type_infeatures, type_outfeatures):
        """Dummy set for setting specific inputs and outputs.

        Parameters
        ----------
        type_infeatures: str, optional
            type of the input features.
        type_outfeatures: str, optional
            type of the output features.

        """
        ## Set outformatter as null
        self._out_formatter = null_out_formatter

    ###########################################################################
    ########################## Formatter functions ############################
    ###########################################################################


class DummyDescriptor(BaseDescriptorModel):
    """Dummy descriptor model created for testing purposes."""

    name_desc = "Dummy descriptor"

    def __init__(self):
        self._out_formatter = null_out_formatter
        ## Check descriptormodel
        self._assert_correctness()

    def compute(self, pointfeats, point_pos):
        """From [iss][nei][feats] to [iss][feats].

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
        return [pfeats[0] for pfeats in pointfeats]

    def set_functions(self, type_infeatures, type_outfeatures):
        """Set functions we are going to use taking into account the type
        of inputs we are going to receive or output.

        Parameters
        ----------
        type_infeatures: str, optional
            type of the input features.
        type_outfeatures: str, optional
            type of the output features.

        """
        if type_infeatures == 'ndarray':
            self._f_default_names = array_featurenames
        else:
            self._f_default_names = list_featurenames


class NullPhantomDescriptor(BaseDescriptorModel):
    """Dummy descriptor model created for testing purposes."""

    name_desc = "Dummy descriptor"

    def __init__(self):
        self._out_formatter = null_out_formatter
        ## Check descriptormodel
        self._assert_correctness()

    def compute(self, pointfeats, point_pos):
        """From [iss][nei][feats] to [iss][feats]

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
        return [pfeats[0] for pfeats in pointfeats]

    def set_functions(self, type_infeatures, type_outfeatures):
        """Set functions we are going to use taking into account the type
        of inputs we are going to receive or output.

        Parameters
        ----------
        type_infeatures: str, optional
            type of the input features.
        type_outfeatures: str, optional
            type of the output features.

        """
        self._f_default_names = lambda x: [0]


class GeneralDescriptor(BaseDescriptorModel):
    """General descriptor model totally personalizable.
    """

    name_desc = "General descriptor"

    def __init__(self, characterizer, completer=None, out_formatter=None,
                 featurenames=None):
        """A general descriptor container. You can set the class by giving it
        the functions needed to work.

        Parameters
        ----------
        characterizer: function
            the function which acts as the core-characterizer.
        completer: function
            the function which corrects the final output of the whole measure.
        out_formatter: function
            the function which formats the output of the descriptors computed
            in the core-characterizer function.
        featurenames: function
            the function which creates or extracts the names of the features
            from the features data.

        """
        ## Specific class settings
        self.compute = characterizer
        if completer is not None:
            self.to_complete_measure = completer
        self._format_extra_functions(out_formatter, featurenames)
        self._assert_correctness()

    def _format_extra_functions(self, out_formatter, featurenames):
        """Internal function to set some function into that class.

        Parameters
        ----------
        out_formatter: function
            the function which formats the output of the descriptors computed
            in the core-characterizer function.
        featurenames: function
            the function which creates or extracts the names of the features
            from the features data.

        """
        ## Initial function set
        if out_formatter is None:
            self._out_formatter = null_out_formatter
        else:
            self._out_formatter = out_formatter
        if featurenames is None:
            self._f_default_names = array_featurenames
        else:
            self._f_default_names = featurenames


class Interpolator(BaseDescriptorModel):
    """The interpolator container. It is the minimum container for
    interpolation tasks.

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
        """The interpolator container. It is the minimum container for
        interpolation tasks.

        Parameters
        ----------
        f_weight: function
            the function which defines the weights from distances.
        pars_w: dict
            the parameters of the weights definition from distances.
        f_dens: function
            the function definition of the density assignation kernel.
        pars_d: dict
            the parameters for assigning the kernel.

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
