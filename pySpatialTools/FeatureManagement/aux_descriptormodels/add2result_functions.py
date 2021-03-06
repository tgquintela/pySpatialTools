
"""
add2results functions
---------------------
This module contains different add2results functions in order to be used
by the different descriptor models we code or are coded in the module.
This is a compulsary function placed in the descriptor model which is used by
the


This function is a compulsary function in the descriptor model object in
order to be passed to the spatial descriptor model.

** Main properties:
   ---------------
INPUTS:
- aggdescriptors_idxs: the features associated directly to each aggregation.
    The could be expressed in list, np.ndarray or dict.

OUTPUTS:
- descriptors: in dict or array format depending on the descriptors convenience

"""


def sum_addresult_function(x, x_i, vals_i):
    """Sum the result to the final result.

    Parameters
    ----------
    x: list or np.ndarray
        the stored measure. The standard input is:
            * x: (vals_i, feats, ks)
    x_i: list [ks](feats) or np.ndarray (ks, feats)
        the parcial measure to be stored.
    vals_i: list or np.ndarray
        the information of the stored indice.

    Returns
    -------
    x: np.ndarray (vals_i, feats, ks)
        the stored measure.

    """
    for k in range(len(vals_i)):
        x[[vals_i[k]], :, k] += x_i[k]
    return x


def append_addresult_function(x, x_i, vals_i):
    """Append the result to the final result.

    Parameters
    ----------
    x: list
        the stored measure. The standard input is:
            * x: [ks][iss_vals]{feats}
    x_i: list
        the parcial measure to be stored.
    vals_i: list or np.ndarray
        the information of the stored indice.

    Returns
    -------
    x: list
        the stored measure.

    """
    for k in range(len(vals_i)):
        for i in range(len(vals_i[k])):
            if type(x[k][vals_i[k][i]]) == list:
                x[k][vals_i[k][i]].append(x_i[k][i])
            else:
                # If precollapsed
                x[k][vals_i[k][i]] = [x[k][vals_i[k][i]]]
                x[k][vals_i[k][i]].append(x_i[k][i])
    return x


def replacelist_addresult_function(x, x_i, vals_i):
    """Replace the element in a preinitialized list. For unknown vals_i.

    Parameters
    ----------
    x: list
        the stored measure. The standard input is:
            * x: [ks][0][iss_vals]{feats} and [ks][1][iss_vals](vals_i)
    x_i: list
        the parcial measure to be stored.
    vals_i: list or np.ndarray
        the information of the stored indice.

    Returns
    -------
    x: list
        the stored measure.

    See also:
    ---------
    sparse_dict_completer

    """
    ## Adding to result
    for k in range(len(vals_i)):
        x[k][0].append(x_i[k])
        x[k][1].append(vals_i[k])
    return x
