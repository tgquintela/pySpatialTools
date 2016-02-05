
"""
Auxiliary functions
-------------------
Functions to perform general computations of statistics or transformations
useful for compute the models.


TODO:
-----
Preprocess utils?
Statistical utils?

"""

import numpy as np


###############################################################################
############################# Auxiliary functions #############################
###############################################################################
def compute_global_counts(df, type_vars):
    """Compute counts of each values. It is useful for discrete or categorical
    variables.

    Parameters
    ----------
    df: pandas.DataFrame
        the whole dataset.
    type_vars: list
        list of the possible variables in which we are interested to study.

    Returns
    -------
    N_x: dict of numpy.ndarrays
        the counts of values of the variables for each variable.
    type_vals: dict of lists
        the lists of possible values for each variable.

    """
    N_x = {}
    type_vals = {}
    for var in type_vars:
        t_vals = sorted(list(df[var].unique()))
        aux_nx = [np.sum(df[var] == type_v) for type_v in t_vals]
        aux_nx = np.array(aux_nx)
        N_x[var], type_vals[var] = aux_nx, t_vals
    return N_x, type_vals


def mapping_typearr(type_arr, type_vars):
    """Function to map the values of a categorical variables to an integer map.

    Parameters
    ----------
    type_arr: numpy.ndarray, shape (N, m)
        the values of a categorical variables.
    type_vars: list, len (m)
        the list of categorical variables we want to map.

    Returns
    -------
    maps: dict or numpy.ndarrays
        the maps between the categorical values and the assigned integer
        values.

    """
    maps = {}
    for i in range(type_arr.shape[1]):
        vals = np.unique(type_arr[:, i])
        maps[type_vars[i]] = dict(zip(vals, range(vals.shape[0])))
    return maps


def generate_replace(type_vals):
    """Generate the replace for using indices (numerical indices and not str
    indices) and to save memory.

    Parameters
    ----------
    type_vals: dict of lists
        the lists of possible values for each variable.

    Returns
    -------
    repl: dict
        the replace dictionary in which we map each variable name to a numeric
        integer value.

    """
    repl = {}
    for v in type_vals.keys():
        repl[v] = dict(zip(type_vals[v], range(len(type_vals[v]))))
    return repl
