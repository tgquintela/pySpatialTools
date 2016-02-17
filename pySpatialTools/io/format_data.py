
"""
Format data
-----------
Module which groups all the possible transformation of variables to make them
useful in the framework provided by this package.

"""
import numpy as np
import multiprocessing as mp
from pySpatialTools.utils.transformation_utils import split_df,\
    compute_reindices
import numpy as np


def format_input(df, typevars, reindices=None):
    locs, feat_arr, info_ret, cond_agg = split_df(df, self.typevars)
    reindices = compute_reindices(df, reindices)
    self.sp_descriptormodel.set(locs, feat_arr, info_ret, cond_agg, reindices)
    # clean unnecessary
    del df, locs, feat_arr, info_ret, cond_agg, reindices
    pass
