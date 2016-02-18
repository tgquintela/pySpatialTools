
"""
Distribution of categorical variables can be plotted and inspected with the
functions of this module.
"""

import matplotlib.pyplot as plt
import numpy as np


def barplot_plot(x, logscale=False):
    """Function to explore distribution of a categorical variable.
    """

    ## 0. Setting needed variables
    counts = x.value_counts()
    v = list(counts.index)
    c = np.array(counts)
    x = np.arange(len(v))

    ## 1. Plotting
    fig = plt.figure()
    ax0 = plt.subplot()
    ax0.bar(x, c, align='center')
    ax0.set_xlim([-0.4, len(v)])
    if logscale:
        ax0.set_yscale('log')

    ## 2. Making up the plot
    # Ticks management
    if len(v) < 100:
        ax0.set_xticks(x)
        ax0.set_xticklabels(v)
        plt.setp(ax0.xaxis.get_majorticklabels(), rotation=310)
    else:
        ax0.xaxis.set_ticklabels([])
    # Axes labels management
    plt.xlabel('Value')
    plt.ylabel('Counts')
    # Title
    fig.suptitle('Distribution exploration of a categorical variable',
                 fontsize=14, fontweight='bold')

    return fig
