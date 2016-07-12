
"""
Completer functions
-------------------
This module contain possible functions to complete the final measure.

"""

import numpy as np
from scipy.sparse import coo_matrix
from out_formatters import count_out_formatter_dict2array


def sparse_dict_completer(measure, global_info=None):
    """Sparse completer transform the dictionaries into a sparse matrices.

    See also:
    ---------
    append_addresult_function

    """
    pos_feats = []
    n_iss = len(measure[0])
    for k in range(len(measure)):
        for vals_i in range(len(measure[k])):
            ## Collapse into one dictionary
            # Just collapsed
            if type(measure[k][vals_i]) == dict:
                continue
            # Add dictionaries and possible features
            d = {}
            for i in range(len(measure[k][vals_i])):
                aux_dict = measure[k][vals_i][i]
                aux_keys = aux_dict.keys()
                pos_feats += aux_keys
                for e in aux_keys:
                    if e in d:
                        d[e] += aux_dict[e]
                    else:
                        d[e] = aux_dict[e]
            measure[k][vals_i] = d
    ## Collapsing
    feats_names = list(np.unique(pos_feats))
    for k in range(len(measure)):
        data, iss, jss = [], [], []
        for i in range(len(measure[k])):
            aux_jss = measure[k][i].keys()
            jss += [feats_names.index(e) for e in aux_jss]
            data += measure[k][i].values()
            iss += len(measure[k][i])*[i]

        ## Building the matrix and storing it in measure
        shape = (n_iss, len(feats_names))
        data, iss, jss = np.array(data), np.array(iss), np.array(jss)
        measure[k] = coo_matrix((data, (iss, jss)), shape=shape)

#    measure = np.array(measure)
    return measure


def sparse_dict_completer_unknown(measure, global_info=None):
    """Sparse completer transform the dictionaries into a sparse matrices.

    See also:
    ---------
    replacelist_addresult_function

    """
    ## Completing measure
    for k in range(len(measure)):
        data, iss, jss = [], [], []
        val_res_con = np.concatenate(measure[k][1])
        if len(np.unique(val_res_con)) == len(val_res_con):
            for i in range(len(measure[k][1])):
                jss += measure[k][0][i].keys()
                data += measure[k][0][i].values()
                iss += len(measure[k][0][i])*[measure[k][1][i]]
        else:
            for i in range(len(measure[k][1])):
                dicti = {}
                for v in range(len(measure[k][1][i])):
                    keys = measure[k][0][i][v].keys()
                    values = measure[k][0][i][v].values()
                    for j in xrange(len(keys)):
                        if keys[j] in dicti:
                            dicti[keys[j]] += values[j]
                        else:
                            dicti[keys[j]] = values[j]
#                        try:
#                            dicti[keys[j]] += values[j]
#                        except:
#                            dicti[keys[j]] = values[j]
                    ### WARNING: TODO: Featurenames transformation into numbers
                    jss += [int(e) for e in dicti.keys()]
                    iss += [measure[k][1][i][v]]*len(dicti.keys())
                    data += dicti.values()
        ## Building the matrix and storing it in measure
        shape = (int(np.max(iss))+1, int(np.max(jss))+1)
        data, iss, jss = np.array(data), np.array(iss), np.array(jss)
        measure[k] = coo_matrix((data, (iss, jss)), shape=shape)

    return measure


def null_completer(measure, global_info=None):
    "Do not change the measure."
    return measure


def null_completer_concatenator(measure, global_info=None):
    """Don not change anything, only concatenate all the partial descriptors
    for each vals_i."""
    return np.concatenate(measure)


def weighted_completer(measure, global_info):
    """Weight the different results using the global info.
    It is REQUIRED that the global_info is an array of the same length as the
    measure.
    """
    if global_info is None:
        return measure
    if len(global_info.shape) != 3:
        global_info = global_info.ravel()
        assert len(measure) == len(global_info)
        global_info = global_info.reshape((len(global_info), 1, 1))
    measure = np.multiply(measure, global_info)
    return measure
