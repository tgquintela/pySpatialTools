
"""
General descriptormodels
------------------------
Module which contains general useful definitions of descriptor models.

"""

from descriptormodel import DescriptorModel
## Specific functions
from pySpatialTools.Feature_engineering.aux_descriptormodels import\
    sum_addresult_function, null_completer, array_featurenames,\
    null_out_formatter


class DescriptorGeneral(DescriptorModel):
    """General descriptor model which is able to construct any type of
    descriptor model which this framework is able to build.
    """
    name_desc = "General spatial descriptor"
    _n = 0
    _nullvalue = 0

    def __init__(self, features, compute_characs, reducer, aggdescriptor,
                 to_complete_measure=None, sp_typemodel='matrix',
                 name_desc=""):
        """
        Parameters
        ----------
        """
        ## Initial settings
        self._format_parameters(name_desc)
        self._format_input_functions(compute_characs, reducer, aggdescriptor,
                                     to_complete_measure)
        ## Initial function set
        self._out_formatter = null_out_formatter
        self._f_default_names = array_featurenames
        self._defult_add2result = sum_addresult_function
        ## Format features
        self._format_features(features)
        ## Type of built result
        self._format_map_vals_i(sp_typemodel)
        ## Format function to external interaction and building results
        self._format_result_building()

    def _format_input_functions(self, compute_characs, reducer, aggdescriptor,
                                to_complete_measure):
        "Formatter function for the input functions."
        ## Compute characs
        self.compute_characs = compute_characs
        ## Reducer
        self.reducer = reducer
        ## Aggdescriptor setting
        self.aggdescriptor = aggdescriptor
        ## Completer setting (global info here?)
        if to_complete_measure is None:
            self.to_complete_measure = null_completer
        else:
            self.to_complete_measure = to_complete_measure

    def _format_parameters(self, name_desc):
        "Format patameters input."
        if name_desc != "":
            self.name_desc = name_desc
