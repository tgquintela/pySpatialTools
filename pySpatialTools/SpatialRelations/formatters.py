
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
        for i in range(relations.shape[0]):
            relations_o.append(list(relations.getrow(i).nonzero()[0]))
    return relations_o


def _relations_parsing_creation(relations_info):
    """Function which uniforms the relations info to be useful in other
    parts of the code.

    Standarts
    * relations object
    * (main_relations_info, pars_rel)
    * (main_relations_info, pars_rel, _data)
    * (main_relations_info, pars_rel, _data, data_in)
    """
    if isinstance(relations_info, RegionDistances):
        pass
    elif type(relations_info) != tuple:
        relations_info = RegionDistances(relations_info)
    else:
        assert(type(relations_info) == tuple)
        assert(len(relations_info) in [2, 3, 4])
        pars_rel = relations_info[1]
        if len(relations_info) == 4:
            pars_rel['data_in'] = relations_info[3]
        if len(relations_info) >= 3:
            pars_rel['_data'] = relations_info[2]
        relations_info = RegionDistances(relations_info[0], **pars_rel)
    return relations_info
