
"""
Auxiliary functions
-------------------
Functions to perform general computations of statistics or transformations
useful for compute the models.

"""

import numpy as np


###############################################################################
############################# Auxiliary functions #############################
###############################################################################
def init_compl_arrays(df, typevars, reindices):
    """Auxiliary function to prepare the initialization and preprocess of the
    required input variables in order to format them properly.
    """
    N_t = df.shape[0]

    # Reindices creation
    if type(reindices) != np.ndarray:
        reindices = reindices_creation(df, reindices)

    # Extract arrays
    locs = df[typevars['loc_vars']].as_matrix()
    ndim = len(locs.shape)
    locs = locs if ndim > 1 else locs.reshape((N_t, 1))

    feat_arr = df[typevars['feat_vars']].as_matrix()
    ndim = len(feat_arr.shape)
    feat_arr = feat_arr if ndim > 1 else feat_arr.reshape((N_t, 1))

    info_ret = df[typevars['info_ret']].as_matrix()
    cond_agg = df[typevars['cond_agg']].as_matrix()

    return locs, feat_arr, info_ret, cond_agg, reindices


def reindices_creation(df, permuts):
    "Function to create reindices for permuting elements of the array."
    N_t = df.shape[0]
    reindex = np.array(df.index)
    reindex = reindex.reshape((N_t, 1))
    if permuts is not None:
        if type(permuts) == int:
            if permuts != 0:
                permuts = [np.random.permutation(N_t) for i in range(permuts)]
                permuts = np.vstack(permuts).T
                bool_ch = len(permuts.shape) == 1
                permuts = permuts.reshape((N_t, 1)) if bool_ch else permuts
        elif type(permuts) == np.ndarray:
            n_per = permuts.shape[1]
            permuts = [reindex[permuts[:, i]] for i in range(n_per)]
            permuts = np.hstack(permuts)
    bool_ch = permuts is None or permuts == 0
    reindex = [reindex] if bool_ch else [reindex, permuts]
    reindices = np.hstack(reindex)
    return reindices


def compute_global_counts(df, type_vars):
    "Compute counts of each values."
    N_x = {}
    type_vals = {}
    for var in type_vars:
        t_vals = sorted(list(df[var].unique()))
        aux_nx = [np.sum(df[var] == type_v) for type_v in t_vals]
        aux_nx = np.array(aux_nx)
        N_x[var], type_vals[var] = aux_nx, t_vals
    return N_x, type_vals


def mapping_typearr(type_arr, type_vars):
    maps = {}
    for i in range(type_arr.shape[1]):
        vals = np.unique(type_arr[:, i])
        maps[type_vars[i]] = dict(zip(vals, range(vals.shape[0])))
    return maps


def generate_replace(type_vals):
    "Generate the replace for use indices and save memory."
    repl = {}
    for v in type_vals.keys():
        repl[v] = dict(zip(type_vals[v], range(len(type_vals[v]))))
    return repl
