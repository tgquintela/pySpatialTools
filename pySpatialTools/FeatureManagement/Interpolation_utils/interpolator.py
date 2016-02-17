
"""

TODO
----
- implementation of 3 cases:
    - autoretrieve
    - non-autoretrieve without features_i
    - non-autoretrieve with features_i
"""


from ..descriptormodel import DescriptorModel

## Specific functions
from ..aux_descriptormodels import\
    sum_reducer, sum_addresult_function, array_featurenames,\
    null_out_formatter


class Interpolator(DescriptorModel):
    """
    TODO
    ----
    Matrix of weights between distances and own properties.
    (Weights not depending only on distances but also on ij type)
    Integrate special parameters in the functions with function-creators.

    """

    name_desc = "Density assignation descriptor"

    def __init__(self, weighting_avg, features, sp_typemodel='matrix'):
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
        self.compute_characs = weighting_avg
        self.reducer = sum_reducer
        self.aggdescriptor = weighting_avg
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
