
"""
Syntetic data
-------------
Module which groups useful syntetic data to use this library. It also helps to
format the data to the proper format to use this library.

"""

import numpy as np


def create_syntetic_data(n, m_agg, m_feat):
    "Creation of a dataset with random syntetic data."
    typevars = {'loc_vars': ['x', 'y'], 'feat_vars': ['feat_a'],
                'agg_var': 'agg_var'}
    locs = np.random.random((n, 2))
    agg_arr = np.random.randint(0, m_agg, n)
    feat_arr = np.random.randint(0, m_feat, n).reshape(n, 1)
    locs = pd.DataFrame(locs, columns=typevars['loc_vars'])
    agg_arr = pd.DataFrame(agg_arr, columns=typevars['agg_var'])
    feat_arr = pd.DataFrame(feat_arr, columns=typevars['feat_vars'])
    df = pd.concat([locs, agg_arr, feat_arr], axis=1)

    return df, typevars


def create_reindices(n, m):
    "Creation of reindices for a permutation."
    reindices = np.zeros((n, m+1)).astype(int)
    reindices[:, 0] = np.arange(n).astype(int)
    for i in range(m):
        reindices[:, i] = np.random.permutation(np.arange(n).astype(int))
    return reindices


def format_info_ret(df, typevars, info_ret):
    "Function to format and integrate the info_ret to the data."
    if type(info_ret) == str:
        typevars['info_ret'] = info_ret
    elif type(info_ret) == np.ndarray:
        info_ret = info_ret
        df['info_ret'] = info_ret
        typevars['info_ret'] = 'info_ret'
    elif type(info_ret) in [int, float, bool]:
        info_ret = (np.ones(df.shape[0])*info_ret).astype(type(info_ret))
        df['info_ret'] = info_ret
        typevars['info_ret'] = 'info_ret'

    return df, typevars


def format_cond_agg(df, typevars, cond_agg):
    "Function to format and integrate the cond_agg to the data."
    if type(cond_agg) == str:
        cond_agg = df[cond_agg].as_matrix()
        typevars['cond_agg'] = cond_agg
    elif type(cond_agg) == np.ndarray:
        cond_agg = cond_agg
        df['cond_agg'] = cond_agg
        typevars['cond_agg'] = 'cond_agg'
    elif type(cond_agg) in [int, float, bool]:
        cond_agg = (np.ones(df.shape[0])*cond_agg).astype(type(cond_agg))
        df['cond_agg'] = cond_agg
        typevars['cond_agg'] = 'cond_agg'
    return df, typevars
