
"""
This module contains functions which helps in the computation of extra and
complementary data needed and computed from the known data.

TODO:
----
- Other functions not only count.

"""

import numpy as np
import pandas as pd
from format_utils import create_formatted_spdf


###############################################################################
############################ Main functions counts ############################
###############################################################################
def aggregated_counts(agg_arr, feat_arr, reindices, f=None):
    df, typevars = create_formatted_spdf(agg_arr, feat_arr)
    feat_vars, agg_var = typevars['feat_vars'], typevars['agg_var']
    agg_desc, _ = compute_aggregate_counts(df, agg_var, feat_vars, reindices)
    agg_desc = agg_desc[agg_desc.keys()[0]]
    return agg_desc


###############################################################################
############################## Counting functions #############################
###############################################################################
def compute_aggregate_counts(df, agg_var, feat_vars, reindices):
    ## Compute the tables
    agg_values = list(np.unique(df[agg_var]))
    tables = {}
    axis = {}
    for col in feat_vars:
        n_vals = df[col].unique().shape[0]
        aux = np.zeros((len(agg_values), n_vals, reindices.shape[1]))
        for i in range(reindices.shape[1]):
            # The type values
            aux_df = df.loc[:, [agg_var]+feat_vars]
            aux2 = aux_df[feat_vars].reindex(reindices[:, i]).as_matrix()
            aux_df[feat_vars] = aux2
            table, cols = counting_type_by_aggvar(aux_df, agg_var, feat_vars)
            aux[:, :, i] = table.as_matrix()

        tables[col] = aux
        axis[col] = {'rows': agg_values, 'columns': cols}

    return tables, axis


###############################################################################
############################ Auxiliar counts by var ###########################
###############################################################################
def aggregate_by_var(df, agg_var, loc_vars, feat_vars=None):
    """Function to aggregate variables by the selected variable considering a
    properly structured data.
    """
    ## Aggregation
    positions = average_position_by_aggvar(df, agg_var, loc_vars)
    if feat_vars is not None:
        types = aggregate_by_typevar(df, agg_var, feat_vars)
        df_agg, cols = pd.concat([positions, types], axis=1)
        cols = {'types': cols}
        cols['positions'] = list(positions.columns)
    else:
        df_agg = positions
        cols = {'positions': list(positions.columns)}

    return df_agg, cols


def aggregate_by_typevar(df, agg_var, feat_vars):
    "Function to aggregate only by type_var."
    feat_vars = [feat_vars] if type(feat_vars) != list else feat_vars
    df_agg = counting_type_by_aggvar(df, agg_var, feat_vars)
    cols = list(df.columns)
    return df_agg, cols


def average_position_by_aggvar(df, aggvar, loc_vars):
    "Compute the pivot table to assign to cp a geographic coordinates."
    table = df.pivot_table(values=loc_vars, rows=aggvar, aggfunc=np.mean)
    return table


def average_position_by_aggarr(locs, agg_arr):
    "Compute the pivot table to assign to cp a geographic coordinates."
    loc_vars, aggvar = ['x', 'y'], 'agg'
    df = [pd.DataFrame(locs, columns=loc_vars),
          pd.DataFrame(agg_arr, columns=[aggvar])]
    df = pd.concat(df, axis=1)
    table = df.pivot_table(values=loc_vars, rows=aggvar, aggfunc=np.mean)
    return table


def counting_type_by_aggvar(df, aggvar, feat_vars):
    "Compute the counting of types by "
    table = df[[aggvar] + feat_vars].pivot_table(rows=aggvar, cols=feat_vars,
                                                 aggfunc='count')
    table = table.fillna(value=0)
    cols = table.columns.get_level_values(1).unique()
    m = len(cols)
    table = table.loc[:, table.columns[:m]]
    table.columns = cols
    return table, cols


def std_type_by_aggvar(df, aggvar, loc_vars):
    "Compute the counting of types by "
    table = df.pivot_table(rows=aggvar, values=loc_vars,
                           aggfunc=np.std)
    table = table.fillna(value=0)
    table.columns = ['STD-X', 'STD-Y']
    return table


def mean_type_by_aggvar(df, aggvar, loc_vars):
    "Compute the counting of types by "
    table = df.pivot_table(rows=aggvar, values=loc_vars,
                           aggfunc=np.mean)
    table.columns = ['MEAN-X', 'MEAN-Y']
    table = table.fillna(value=0)
    return table


###############################################################################
############################# Auxiliar grid counts ############################
###############################################################################
def computation_aggregate_collapse_i(type_arr, n_vals):
    "Counting the types of each one."
    values = np.unique(type_arr[:, 0])
    counts_i = np.zeros(n_vals[0])
    for j in range(values.shape[0]):
        counts_i[values[j]] = (type_arr == values[j]).sum()
    return counts_i
