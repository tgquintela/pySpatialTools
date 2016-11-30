
"""
Auxiliar resulter building
--------------------------
Auxiliar functions to create resulter building. Resulters have the nex tasks:
* Initialization individuals
* Initialization global
* Adding to result
* Completing

The resulter can be defined by these properties:
* Open-Close limits (if there is a known limitation)
* Array-dict output (if the output is array or dict)
* Selfdriven-Outdriven (if the descriptormodel is the one who creates vals_i
    or it is the map_vals_i). It is defined by `descriptormodel.selfdriven`.

Needed information
* `descriptormodel.selfdriven`: given by the descriptormodel.
* `n_vals_i`: given by the map_vals_i [None, int number]-[Open, close]
* `FeatureManager._out`: given by FeatureManager.
* nfeats: number of features.
* k_perturb: number of perturbations applied.


TODO:
* Not all descriptormodels selfdriven or outdriven

"""

import numpy as np
from aux_descriptormodels import append_addresult_function,\
    replacelist_addresult_function, sparse_dict_completer,\
    sparse_dict_completer_unknown, sum_addresult_function,\
    null_completer, null_completer_concatenator


########################## Collections of functions ###########################
###############################################################################
################### Creation functions initialization_desc ####################
def creation_initialization_desc_array(fm):
    """Creation of the null descriptors in an array-form.

    Parameters
    ----------
    fm: pst.FeatureManager
        the featuresmanager object.

    Returns
    -------
    initialization_desc: function
        the initialization function of null descriptors.

    """
    ## Format initialization descriptors
    def initialization_desc():
        """Initialization of descriptors.

        Returns
        -------
        descriptors: list
            the null descriptors for each perturbation and element.

        """
        descriptors = []
        for i in range(len(fm)):
            aux_i = np.ones((1, len(fm[i].out_features)))
            descriptors.append(aux_i * fm[i]._nullvalue)
        descriptors = np.concatenate(descriptors, axis=1)
        return descriptors
    return initialization_desc


def creation_initialization_desc_dict(fm=None):
    """Creation of the dictionary null descriptors.

    Parameters
    ----------
    fm: pst.FeatureManager (default=None)
        the featuresmanager object.

    Returns
    -------
    initialization_desc: function
        the initialization function of null descriptors.

    """
    ## Format initialization descriptors
    initialization_desc = lambda: [{}]
    return initialization_desc


################## Creation functions initialization_output ###################
def creation_initialization_output_lists(fm):
    """ Format function initialization of global measure of descriptors for
    list of lists result.

    Parameters
    ----------
    fm: pst.FeatureManager
        the featuresmanager object.

    Returns
    -------
    initialization_output: function
        function to initialize the whole complete null measure.

    """
    n_vals_i, _, k_perturb = fm.shape_measure

    def initialization_output():
        return [[[] for i in range(n_vals_i)] for k in range(k_perturb)]
    return initialization_output


def creation_initialization_output_list_selfdriven(fm):
    """ Format function initialization of global measure of descriptors for
    list of lists result.

    Parameters
    ----------
    fm: pst.FeatureManager
        the featuresmanager object.

    Returns
    -------
    initialization_output: function
        function to initialize the whole complete null measure.

    """
    n_vals_i, _, k_perturb = fm.shape_measure

    def initialization_output():
        """Function to initialize the whole complete null measure.

        Returns
        -------
        measure: list
            the measure computed by the whole spatial descriptor model.

        """
        return [[[], []] for k in range(k_perturb)]
    return initialization_output


def creation_initialization_output_list(fm):
    """ Format function initialization of global measure of descriptors for
    list result.

    Parameters
    ----------
    fm: pst.FeatureManager
        the featuresmanager object.

    Returns
    -------
    initialization_output: function
        function to initialize the whole complete null measure.

    """
    initialization_output = lambda: []
    return initialization_output


def creation_initialization_output_closearray(fm):
    """Format function initialization of global measure of descriptors for
    closed array.

    Parameters
    ----------
    fm: pst.FeatureManager
        the featuresmanager object.

    Returns
    -------
    initialization_output: function
        function to initialize the whole complete null measure.

    """
    shape = fm.shape_measure
    assert(all([e is not None for e in shape]))
    initialization_output = lambda: np.zeros(shape)
    return initialization_output


#################### Creation functions _join_descriptors #####################
def creation_null_joiner():
    """Creation of a null joiner of descriptors.

    Returns
    -------
    _join_descriptors: function
        function to join the final descriptors computed by the featuresmanager.

    """
    _join_descriptors = lambda x: x
    return _join_descriptors


def creation_concatenator_joiner():
    """Creation of a concatenator joiner of descriptors.

    Returns
    -------
    _join_descriptors: function
        function to join the final descriptors computed by the featuresmanager.

    """
    _join_descriptors = lambda x: np.concatenate(x)
    return _join_descriptors


################## Default creation functions initialization ##################
def default_creation_initializations(fm):
    """Default creation of initialization.

    Parameters
    ----------
    fm: pst.FeatureManager
        the featuresmanager object.

    Returns
    -------
    init_desc: function
        function to initialize the null descriptor.
    init_output: function
        function to initialize the whole complete null measure.
    _join_descriptors: function
        function to join the final descriptors computed by the
        featuresmanager.
    add2result: function
        function to add the new computed descriptors to the whole measure.
    completer: function
        function to complete the measure once we finish the iterations.

    """
    ## If array descriptors
    if fm._out == 'ndarray':
        init_desc = creation_initialization_desc_array(fm)
        _join_descs = creation_concatenator_joiner()
        # If it is open
        if fm.shape_measure[0] is not None:
            init_output = creation_initialization_output_closearray(fm)
            add2result = sum_addresult_function
            completer = null_completer
        # If it is close
        else:
            init_output = creation_initialization_output_list(fm)
            add2result = append_addresult_function
            completer = null_completer_concatenator
    ## If dict descriptors
    else:
        init_desc = creation_initialization_desc_dict()
        _join_descs = creation_null_joiner()
        # If it is open
        if fm.shape_measure[0] is not None:
            init_output = creation_initialization_output_lists(fm)
            add2result = append_addresult_function
            completer = sparse_dict_completer
        # If it is close
        else:
            init_output =\
                creation_initialization_output_list_selfdriven(fm)
            add2result = replacelist_addresult_function
            completer = sparse_dict_completer_unknown
    return init_desc, init_output, _join_descs, add2result, completer


########################## Collections of resulters ###########################
###############################################################################
class BaseResulter:
    """The basic functions for the base resulter. The resulter is the basic
    information to build the final measure from the computations of the
    featuresmanager.
    """
    def get_functions(self):
        """Function to get all the basic functions which build the resulter."""
        return self.initialization_desc, self.initialization_output,\
            self._join_descriptors,  self.add2result, self.to_complete_measure


class GeneralResulter(BaseResulter):
    """General resulter building object. It contains the main functions to set
    the initialization of the measure and the functions to build it from the
    results of the descriptormodel outputs."""

    def __init__(self, init_desc, init_output, _join_descriptors,  add2result,
                 completer):
        """General resulter instantiation.

        Parameters
        ----------
        init_desc: function
            function to initialize the null descriptor.
        init_output: function
            function to initialize the whole complete null measure.
        _join_descriptors: function
            function to join the final descriptors computed by the
            featuresmanager.
        add2result: function
            function to add the new computed descriptors to the whole measure.
        completer: function
            function to complete the measure once we finish the iterations.

        """
        self.initialization_desc = init_desc
        self.initialization_output = init_output
        self._join_descriptors = _join_descriptors
        self.add2result = add2result
        self.to_complete_measure = completer


class DefaultResulter(BaseResulter):
    """Default resulter building object. It selects and contains the main
    functions to manage descriptors and build the result measure.
    """
    def __init__(self, fm, resulter=None):
        """The default resulter building.

        Parameters
        ----------
        fm: pst.FeatureManager
            the featuresmanager object.
        resulter: pst.BaseResulter or None (default=None)
            the basic information to build the final measure from the
            computations of the featuresmanager.

        """
        if resulter is not None:
            self.initialization_desc = resulter.initialization_desc
            self.initialization_output = resulter.initialization_output
            self._join_descriptors = resulter._join_descriptors
            self.add2result = resulter.add2result
            self.to_complete_measure = resulter.to_complete_measure
        else:
            init_desc, init_output, _join_descs, add2result, completer =\
                default_creation_initializations(fm)
            self.initialization_desc = init_desc
            self.initialization_output = init_output
            self._join_descriptors = _join_descs
            self.add2result = add2result
            self.to_complete_measure = completer
