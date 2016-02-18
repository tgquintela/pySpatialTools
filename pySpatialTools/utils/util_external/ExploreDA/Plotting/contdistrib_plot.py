
"""
Distribution of continious variables can be plotted and inspected with the
functions of this module.
"""

import matplotlib.pyplot as plt
import numpy as np


def cont_distrib_plot(x, n_bins, logscale=False):
    """Function to explore the distribution of a continiuos variable.

    TODO
    ----
    Kernel density estimation

    """
    ## 0. Preparing inputs
    # Filtering nan
    x = x.dropna()
    # Median
    median = x.quantile(0.5)
    x = np.array(x)

    ### A. Plotting
    fig = plt.figure()
    ## 1. Plot histogram
    ax0 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    ax0.hist(x, n_bins)
    # Improving axes
    ax0.set_xlim([x.min(), x.max()])
    ax0.set_ylabel('Counts')
    if logscale:
        ax0.set_yscale('log')
        ax0.set_xscale('log')
    # Mark of median
    l1 = plt.axvline(median, linewidth=2, color='r', label='Median',
                     linestyle='--')
    # Mark of mean
    l2 = plt.axvline(x.mean(), linewidth=2, color='k', label='Mean',
                     linestyle='--')
    ax0.legend([l1, l2], ['Median', 'Mean'])
    ax0.grid(True)

    ## 2. Plot box_plot
    ax1 = plt.subplot2grid((5, 1), (4, 0), sharex=ax0)
    ax1.boxplot(x, 0, 'rs', 0, 0.75)
    # Making up the plot
    mini = x.min()
    maxi = x.max()
    delta = (maxi-mini)/25.
    ax1.set_xlim([mini-delta, maxi+delta])
    if logscale:
        ax1.set_yscale('log')
        ax1.set_xscale('log')
    ax1.grid(True)
    ax1.set_yticklabels('A')
    plt.setp(ax0.get_xticklabels(), visible=False)

    ## 3. Main settings
    fig.suptitle('Distribution exploration of continious variable',
                 fontsize=14, fontweight='bold')
    plt.xlabel('Value')

    return fig
