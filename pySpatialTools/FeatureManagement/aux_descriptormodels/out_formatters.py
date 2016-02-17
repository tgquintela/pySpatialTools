
"""
out formatters
--------------
This module contain the different functions to format the output.
"""

import numpy as np


def count_out_formatter(feats, out_features, _out, _nullvalue):
    """Function which contains the out formatting. Has to deal with aggregated
    and point data."""
    ## Correct format
    if type(feats).__name__ == _out:
        return feats
    ## Change the format

    if type(feats) == dict:
        # so _out == ndarray
        feats_o = np.ones(len(out_features))*_nullvalue
        for e in feats:
            feats_o[list(out_features).index(str(e))] = feats[e]
        if len(feats_o.shape) == 1:
            feats_o = feats_o.reshape((1, feats_o.shape[0]))
    elif type(feats) == np.ndarray:
        # so _out == dict
        feats_o = dict(zip(out_features, feats.ravel()))

    try:
        if type(feats) == dict:
            # so _out == ndarray
            feats_o = np.ones(len(out_features))*_nullvalue
            for e in feats:
                feats_o[list(out_features).index(str(e))] = feats[e]
            if len(feats_o.shape) == 1:
                feats_o = feats_o.reshape((1, feats_o.shape[0]))
        elif type(feats) == np.ndarray:
            # so _out == dict
            feats_o = dict(zip(out_features, feats.ravel()))
    except:
        raise Exception("Incorrect _out format.")

    return feats_o


def null_out_formatter(feats, out_features, _out, _nullvalue):
    """Function which contains the out formatting. It has to deal with
    aggregated and point data."""

    return feats
