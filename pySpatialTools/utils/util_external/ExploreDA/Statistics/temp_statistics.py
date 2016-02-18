
"""
Module which groups all the functions related with the computing of statistics
of temporal variables.

TODO
----
Not predefined date range.
"""

import numpy as np
import datetime
from ..Plotting import general_plot


def compute_temp_describe(df, info_var):
    """"""

    ## 0. Needed variables
    if info_var['variables'] == list:
        variable = info_var['variables'][0]
    else:
        variable = info_var['variables']

    ## 1. Summary computation
    summary = info_var
    summary['pre_post'] = count_temp_stats(df[variable],
                                           ['2006-01-01', '2012-12-31'],
                                           ['pre', 'through', 'post'])
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)
    return summary


def count_temp_stats(tmps, date_ranges, tags=None):
    """Function used to compute stats of a temporal var."""

    ## 0. Variable needed
    mini = tmps.min()
    maxi = tmps.max()
    date_ranges = [mini]+date_ranges+[maxi]
    for i in range(len(date_ranges)):
        if type(date_ranges[i]) == str:
            aux = datetime.datetime.strptime(date_ranges[i], '%Y-%m-%d')
            date_ranges[i] = aux
    if tags is None:
        tags = [str(e) for e in range(len(date_ranges)-1)]
    n_rang = len(tags)

    ## 1. Counting
    counts = {}
    for i in range(n_rang):
        counts[tags[i]] = np.logical_and(tmps >= date_ranges[i],
                                         tmps <= date_ranges[i+1]).sum()

    return counts
