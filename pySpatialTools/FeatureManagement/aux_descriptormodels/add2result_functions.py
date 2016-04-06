
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
    "Sum the result to the final result."
    for k in range(len(vals_i)):
        x[[vals_i[k]], :, k] += x_i[k]
    return x


def append_addresult_function(x, x_i, vals_i):
    """Append the result to the final result.
    * x: [ks][iss_vals]{feats}
    """
    for k in range(len(vals_i)):
        for i in range(len(vals_i[k])):
            x[k][vals_i[k][i]].append(x_i[k][i])
    return x


def replacelist_addresult_function(x, x_i, vals_i):
    """Replace the element in a preinitialized list. For unknown vals_i.
    * x: [ks][iss_vals][0]{feats} and [ks][iss_vals][1](vals_i)

    See also:
    ---------
    sparse_dict_completer

    """
    ## Adding to result
    for k in range(len(vals_i)):
        x[k][0].append(x_i[k])
        x[k][1].append(vals_i[k])
    return x
