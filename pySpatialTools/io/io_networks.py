
"""
io_networks
-----------
Module to export networks.

"""

from scipy.sparse import coo_matrix, issparse


def scipy_format_networkx(net, tags, autotags=None):
    """Function to format a network in sparse format to networkx format.

    Parameters
    ----------
    net: scipy.sparse
        representation of the relation in a scipy sparse way.
    tags: list
        the name of the nodes.
    autotags: list or None
        the name of the second group of nodes. If there is not a change between
        in-nodes and out-nodes it is None.

    Returns
    -------
    netx: networkx.Graph
        a networkx representation of the net.

    """

    if issparse(net):
        raise TypeError("`net` is not sparse.")
