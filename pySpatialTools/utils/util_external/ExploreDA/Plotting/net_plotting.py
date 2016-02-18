
"""

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_net_distribution(net_mat, n_bins):
    """"""
    net_mat = net_mat.reshape(-1)

    fig = plt.figure()
    plt.hist(net_mat, n_bins)

    l1 = plt.axvline(net_mat.mean(), linewidth=2, color='k', label='Mean',
                     linestyle='--')
    plt.legend([l1], ['Mean'])

    return fig


def plot_heat_net(net_mat, sectors):
    """"""
    vmax = np.sort([np.abs(net_mat.max()), np.abs(net_mat.min())])[::-1][0]
    n_sectors = len(sectors)

    fig = plt.figure()
    plt.imshow(net_mat, interpolation='none', cmap=plt.cm.RdYlGn,
               vmin=-vmax, vmax=vmax)
    plt.xticks(range(n_sectors), sectors)
    plt.yticks(range(n_sectors), sectors)
    plt.xticks(rotation=90)
    plt.colorbar()
    return fig
