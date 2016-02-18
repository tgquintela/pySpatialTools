
"""
Temporal plots to understand the temporal structure of the data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
from dateutil.relativedelta import relativedelta


def temp_distrib(tmps, agg_time, logscale=False):
    """

    Parameters
    ----------
    tmps: pandas.Series
        The datetimes to consider.
    agg_time: str, optional
        The time of which make the aggregation. The options are:
        ['year', 'month','week','day', 'hour', 'minute', 'second',
        'microsecond', 'nanosecond']

    """

    ## 0. Time aggregation
    tmps = tmps.dropna()
    # Vector of times range
    mini = tmps.min()
    maxi = tmps.max()
    delta = relativedelta(**{agg_time+'s': 1})
    ranges = range_time(mini, maxi+delta, agg_time)
    ranges = [strformat(t, agg_time) for t in ranges]
    nt = len(ranges)
    # Format as strings
    tmps = tmps.apply(lambda x: strformat(x, agg_time))
    ## 1. Counts
    counts = tmps.value_counts()
    count_values = []
    for t in ranges:
        if t in list(counts.index):
            count_values.append(counts[t])
        else:
            count_values.append(0)

    ## 2. Plot figure
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_xlim([0, len(ranges)-1])
    ax.bar(range(len(ranges)), count_values, align='center', color='black',
           width=1)
    idxs = ax.get_xticks().astype(int)
    if len(idxs) < 100:
        xticks = [ranges[idxs[i]] for i in range(len(idxs)) if idxs[i] < nt]
        ax.set_xticklabels(xticks)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=310)
    if logscale:
        ax.set_yscale('log')

    return fig


def temp_distrib_agg(tmps, agg_time, total_agg=False):
    """
    total_agg: boolean
        If true, only consider the level of aggregation selected in agg_time,
        else it is considered all the superiors levels up to the agg_time.
    """
    pass


def distrib_across_temp(df, variables, times=None, ylabel='', logscale=False):
    """Function to plot """

    ## 0. Preparing times
    if times is None:
        times = np.arange(2006, 2006+len(variables))
    times = [str(t) for t in times]

    mini = df[variables].min().min()
    maxi = df[variables].max().max()
    delta = (maxi-mini)/25.

    ## 1. Plotting
    fig = plt.figure()
    ax = plt.subplot()
    df[variables].boxplot()
    # Making up the plot
    plt.xlabel('Years')
    if logscale:
        ax.set_yscale('log')
    ax.set_ylim([mini-delta, maxi+delta])
    ax.grid(True)
    ax.set_xticklabels(times)
    plt.ylabel(ylabel)
    plt.title("Distribution by years")

    return fig


###############################################################################
############################# AUXILIARY FUNCTIONS #############################
###############################################################################
def time_aggregator(time, agg_time):
    """Transform a datetime to another with the level of aggregation selected.

    time: datetime object
        the datetime information
    agg_time: str, optional
        The time of which make the aggregation. The options are:
        ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']

    TODO
    ----
    Add week:
        ['year', 'month','week','day', 'hour', 'minute', 'second',
        'microsecond']

    Notes
    -----
    Probably DEPRECATED. Better use strformat for current needs.

    """

    ## 1. Preparation for the aggregation
    agg_ts_list = ['year', 'month', 'day', 'hour', 'minute', 'second',
                   'microsecond']
    idx = agg_ts_list.index(agg_time)+1
    timevalues = list(time.timetuple())

    ## 2. Application of the aggregation
    datedict = dict(zip(agg_ts_list[:idx], timevalues[:idx]))
    if idx < 3:
        datedict['day'] = 1
    if idx < 2:
        datedict['month'] = 1
    time = datetime.datetime(**datedict)
    return time


def range_time(time0, time1, agg_time, increment=1):
    """Produce all the possible times at that level of aggregation between the
    selected initial and end time.
    """
    curr = time0
    delta = relativedelta(**{agg_time+'s': increment})
    range_times = []
    while curr <= time1:
        range_times.append(curr)
        curr += delta
    return range_times


def strformat(time, agg_time, format_mode='name'):
    """Transform a datetime to a string with a level of aggregation desired.
    Now only format to a day aggregation.
    """
    ## 0. Preparing needed variables
    if format_mode == 'name':
        format_list = ['%Y', '%b', '%d']
    elif format_mode == 'number':
        format_list = ['%Y', '%m', '%d']
    agg_ts_list = ['year', 'month', 'day']
    idx = agg_ts_list.index(agg_time)+1
    format_list = format_list[:idx]
    ## 1. Formatting
    format_str = '-'.join(format_list)
    time_str = time.strftime(format_str)
    return time_str
