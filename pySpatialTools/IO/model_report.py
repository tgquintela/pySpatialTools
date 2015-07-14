
"""
Module used to group the functions and utils to built a report from a model
application.
"""

from Mscthesis.Plotting.net_plotting import plot_net_distribution,\
    plot_heat_net
from os.path import exists, join
from os import makedirs


def create_model_report(net, sectors, dirname, reportname):
    "Creation of a report for the model applied."

    # Check and create the folders
    if not exists(dirname):
        makedirs(dirname)
    if not exists(join(dirname, reportname)):
        makedirs(join(dirname, reportname))
    if not exists(join(join(dirname, reportname), 'Images')):
        makedirs(join(join(dirname, reportname), 'Images'))

    # Creation of the plots
    fig1 = plot_net_distribution(net, 50)
    fig2 = plot_heat_net(net, sectors)

    fig1.savefig(join(join(join(dirname, reportname), 'Images'), 'net_hist'))
    fig2.savefig(join(join(join(dirname, reportname), 'Images'), 'heat_net'))

    return fig1, fig2
