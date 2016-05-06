
"""
Feature names
-------------
Functions related with the computation of the names of the variables if they
are not given in the Feature object when it is instantiated.

Parameters
----------
features_o: np.ndarray, (iss, feats) or list of dicts, [iss]{feats}
    the features collection for one ks.

Returns
-------
featurenames: list
    the list of featurenames.

"""

import numpy as np


def counter_featurenames(features_o):
    """Compute default feature names of variables when there is a variable
    categorical used in a counter type descriptor."""
    if type(features_o) == np.ndarray:
        featurenames = list(np.unique(features_o[:, 0]))
    else:
        try:
            #featurenames = list(np.unique(features_o.features[:, 0]))
            featurenames = list_featurenames(features_o)
        except:
            msg = "Incorrect feature type input in order to compute its "
            msg += "features names."
            raise TypeError(msg)
    featurenames = [str(int(e)) for e in featurenames]
    return featurenames


def general_featurenames(features_o):
    """Compute featurenames of the array-like features and let as void
    featurenames for the dict-based featues."""
    if type(features_o) == list and type(features_o[0]) == list:
        featurenames = []
    else:
        featurenames = list(np.arange(len(features_o[0])))
    return featurenames


def list_featurenames(features_o):
    """Compute the featurenames for list of dicts features collections."""
    keys = []
    for i in range(len(features_o)):
        keys += features_o[i].keys()
    featurenames = list(set(keys))
    return featurenames


def array_featurenames(features_o):
    "Compute default feature names of variables when there is an array type."
    if type(features_o) == np.ndarray:
        featurenames = list(np.arange(features_o.shape[1]))
    else:
        if type(features_o[0]) == np.ndarray:
            featurenames = list(np.arange(features_o.shape[1]))
        else:
            msg = "Incorrect feature type input in order to compute its "
            msg += "features names."
            raise TypeError(msg)
    return featurenames
