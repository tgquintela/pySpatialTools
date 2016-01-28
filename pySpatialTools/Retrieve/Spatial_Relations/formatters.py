
"""
Formatters
----------
Module which contains functions to format outputs relations.

"""

#from scipy.sparse import coo_matrix
import networkx as nx
from regionmetrics import RegionDistances


def format_out_relations(relations, out_):
    """Format relations in the format they is detemined in parameter out_.

    Parameters
    ----------
    relations: scipy.sparse matrix
        the relations expressed in a sparse way.
    out_: optional, ['sparse', 'network', 'sp_relations']
        the output format we desired.

    Returns
    -------
    relations: decided format
        the relations expressed in the decided format.
    """

    if out_ == 'sparse':
        relations_o = relations
    elif out_ == 'network':
        relations_o = nx.from_scipy_sparse_matrix(relations)
    elif out_ == 'sp_relations':
        relations_o = RegionDistances(relations)
    elif out_ == 'list':
        relations_o = []
        for i in relations.shape[0]:
            relations_o.append(list(relations.get_row(i).nonzero()[0]))
    return relations_o
