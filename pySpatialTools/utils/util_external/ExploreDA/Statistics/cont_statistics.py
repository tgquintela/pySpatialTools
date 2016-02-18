
"""
Module which groups all the functions needed to compute the statistics and the
description of the categorical variables.

"""

from ..Plotting import general_plot
import numpy as np
import pandas as pd


## Creation of the table info
def compute_cont_describe(df, info_var):
    """Function created to aggregate all the exploratory information of
    variable studied.
    """

    ## 0. Needed variables
    if info_var['variables'] == list:
        variable = info_var['variables'][0]
    else:
        variable = info_var['variables']

    ## 1. Summary
    summary = info_var
    summary['n_bins'] = 10 if not 'n_bins' in info_var else info_var['n_bins']
    summary['mean'] = df[variable].mean()
    summary['quantiles'] = quantile_compute(df[variable],
                                            summary['n_bins'])
    summary['ranges'] = ranges_compute(df[variable],
                                       summary['n_bins'])
    summary['hist_table'] = cont_count(df, variable,
                                       summary['n_bins'])
    if info_var['logscale'] in [True, 'True', 'TRUE']:
        summary['log_hist_table'] = log_cont_count(df, variable,
                                                   summary['n_bins'])

    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)

    return summary


def quantile_compute(df, n_bins):
    # aux.quantile(np.linspace(0, 1, 11)) # version = 0.15
    quantiles = [df.quantile(q) for q in np.linspace(0, 1, n_bins+1)]
    quantiles = np.array(quantiles)
    return quantiles


def ranges_compute(df, n_bins):
    mini = np.nanmin(np.array(df))
    maxi = np.nanmax(np.array(df))
    ranges = np.linspace(mini, maxi, n_bins+1)
    return ranges


## Continious hist
def cont_count(df, variable, n_bins):
    mini = np.nanmin(np.array(df[variable]))
    maxi = np.nanmax(np.array(df[variable]))
    bins = np.linspace(mini, maxi, n_bins+1)
    labels = [str(i) for i in range(int(n_bins))]
    categories = pd.cut(df[variable], bins, labels=labels)
    categories = pd.Series(np.array(categories)).replace(np.nan, 'NaN')
    counts = categories.value_counts()
    return counts, bins


def log_cont_count(df, variable, n_bins):
    mini = np.nanmin(np.array(df[variable]))
    mini = .001 if mini <= 0 else mini
    maxi = np.nanmax(np.array(df[variable]))
    bins = np.linspace(np.log10(mini), np.log10(maxi), n_bins+1)
    bins = np.power(10, bins)
    bins[0] = np.nanmin(np.array(df[variable]))
    labels = [str(i) for i in range(int(n_bins))]
    categories = pd.cut(df[variable], bins, labels=labels)
    categories = pd.Series(np.array(categories)).replace(np.nan, 'NaN')
    counts = categories.value_counts()
    return counts, bins
