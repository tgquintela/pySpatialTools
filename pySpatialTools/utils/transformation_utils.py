
"""
Transformation Utils
--------------------
Module which groups all the transformations of useful data for this package.

"""

import numpy as np


def split_df(df, typevars):
    """Function for splitting the data in the different types considered in
    the spatial datasets: locations, features, retrieval info, condition of
    aggregation. It will be return the different data in numpy arrays.

    Parameters
    ----------
    df: pandas.DataFrame
        the spatial dataset.
    typevars: dictionary
        the mapping of the different type of variables considered.

    Returns
    -------
    locs, feat_arr, info_ret, cond_agg: numpy.ndarray
        the respective data in numpy.ndarray format.

    """

    N_t = df.shape[0]

    # Extract arrays
    locs = df[typevars['loc_vars']].as_matrix()
    ndim = len(locs.shape)
    locs = locs if ndim > 1 else locs.reshape((N_t, 1))

    feat_arr = df[typevars['feat_vars']].as_matrix()
    ndim = len(feat_arr.shape)
    feat_arr = feat_arr if ndim > 1 else feat_arr.reshape((N_t, 1))

    info_ret = df[typevars['info_ret']].as_matrix()
    cond_agg = df[typevars['cond_agg']].as_matrix()

    return locs, feat_arr, info_ret, cond_agg


def compute_reindices(df, permuts=None):
    """Compute reindices (permutation indices).

    Parameters
    ----------
    df: pandas.DataFrame
        the spatial dataset.
    permuts: int, numpy.ndarray or None
        the information to create the reindices.

    Returns
    -------
    reindices: numpy.ndarray
        the indices of the permutations.

    """

    if type(permuts) == np.ndarray:
        return permuts

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
    bool_ch = permuts is None or permuts == 0
    reindex = [reindex] if bool_ch else [reindex, permuts]
    reindices = np.hstack(reindex)
    return reindices
