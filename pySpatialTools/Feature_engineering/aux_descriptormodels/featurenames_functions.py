
"""
Feature names
-------------
Functions related with the computation of the names of the variables if they
are not given in the Feature object when it is instantiated.

"""

import numpy as np


def counter_featurenames(features_o):
    """Compute default feature names of variables when there is a variable
    categorical used in a counter type descriptor."""
    if type(features_o) == np.ndarray:
        featurenames = list(np.unique(features_o[:, 0]))
    else:
        try:
            featurenames = list(np.unique(features_o.features[:, 0]))
        except:
            msg = "Incorrect feature type input in order to compute its "
            msg += "features names."
            raise TypeError(msg)
    return featurenames


def array_featurenames(features_o):
    "Compute default feature names of variables when there is an array type."
    if type(features_o) == np.ndarray:
        featurenames = list(np.arange(features_o.shape[1]))
    else:
        try:
            featurenames = list(np.arange(features_o.shape[1]))
        except:
            msg = "Incorrect feature type input in order to compute its "
            msg += "features names."
            raise TypeError(msg)
    return featurenames
