
"""
out formatters
--------------
This module contain the different functions to format the output.

Standard inputs:
    Parameters
    ----------
    feats: list of dicts [iss_i]{feats} or np.array (iss_i, feats_n)
        the features information.
    out_features: list of featurenames
        the featurenames.
    _out: optional str ['ndarray', 'dict']
        the type of output desired.
    _nullvalue: float
        the nullvalue desired for a variable.

    Returns
    -------
    feats_o: np.array(iss_i, feats_n) or list of dicts [iss_i]{feats}
        the formatted features.

"""

import numpy as np


def count_out_formatter_general(feats, out_features, _out, _nullvalue):
    """Function which contains the out formatting. Has to deal with aggregated
    and point data.
    Assumption: In counting always are dict.
    """
    ## Correct format
    if _out == 'dict':
        return feats
    elif _out in ['ndarray', 'array']:
        feats_o = count_out_formatter_dict2array(feats, out_features, _out,
                                                 _nullvalue)
    else:
        raise Exception("Incorrect _out format.")
    return feats_o


def count_out_formatter_dict2array(feats, out_features, _out, _nullvalue):
    """Function which contains the out formatting. Has to deal with aggregated
    and point data.
    """
    feats_o = np.ones((len(feats), len(out_features)))*_nullvalue
    print feats, out_features
    for i in range(len(feats)):
        for e in feats[i]:
            feats_o[i, list(out_features).index(str(e))] = feats[i][e]
    return feats_o


def null_out_formatter(feats, out_features, _out, _nullvalue):
    """Function which contains the out formatting. It has to deal with
    aggregated and point data."""

    return feats
