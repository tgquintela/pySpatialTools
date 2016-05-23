
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


#def create_sp_descriptor_points_regs(sp_descriptor, regions_id, elements_i):
#    """"""
#    discretizor, locs, retriever, info_ret, descriptormodel = sp_descriptor
#    retriever = retriever(locs, info_ret, ifdistance=True)
#    loc_r = discretizor.discretize(locs)
#    map_locs = dict(zip(regions_id, elements_i))
#    r_locs = np.array([int(map_locs[r]) for r in loc_r])
#    descriptormodel = descriptormodel(r_locs, sp_typemodel='correlation')
#    sp_descriptor = SpatialDescriptorModel(retriever, descriptormodel)
#    n_e = locs.shape[0]
#    sp_descriptor.reindices = np.arange(n_e).reshape((n_e, 1))
#    return sp_descriptor
#
#
#def create_sp_descriptor_regionlocs(sp_descriptor, regions_id, elements_i):
#    """"""
#    discretizor, locs, retriever, info_ret, descriptormodel = sp_descriptor
#    if type(retriever) == str:
#        regionslocs = discretizor.get_regionslocs()[elements_i, :]
#        return regionslocs, retriever
#
#    # Creation of spdesc model
#    retriever = retriever(discretizor.get_regionslocs()[elements_i, :],
#                          info_ret, ifdistance=True)
#    features = ImplicitFeatures(np.ones(len(elements_i)),
#                                descriptormodel=descriptormodel)
#    featurer = FeaturesManager(features, map_vals_i=elements_i)
#    sp_descriptor = SpatialDescriptorModel(retriever, featurer)
#
#    n_e = len(elements_i)
#    sp_descriptor.reindices = np.arange(n_e).reshape((n_e, 1))
#    return sp_descriptor
#    # Sp descriptor management
#    if type(sp_descriptor) == tuple:
#        activated = sp_descriptor[1] if activated is not None else None
#        regions_id, elements_i = get_regions4distances(sp_descriptor[0],
#                                                       elements, activated)
#        sp_descriptor = create_sp_descriptor_regionlocs(sp_descriptor,
#                                                        regions_id,
#                                                        elements_i)
#        _data = np.array(regions_id)
#        _data = _data.reshape((_data.shape[0], 1))
#    else:
#        regions, elements_i = get_regions4distances(sp_descriptor,
#                                                    elements, activated)
#        _data = np.array(regions)
#        _data = _data.reshape((_data.shape[0], 1))
#
#    ## 1. Computation of relations
#    if type(sp_descriptor) == tuple:
#        relations = pdist(sp_descriptor[0], sp_descriptor[1])
#    else:
#        relations = sp_descriptor.compute_net()[:, :, 0]
#
#
#     # Sp descriptor management
#    if type(sp_descriptor) == tuple:
#        activated = sp_descriptor[1] if activated is not None else None
#        regions_id, elements_i = get_regions4distances(sp_descriptor[0],
#                                                       elements, activated)
#        sp_descriptor = create_sp_descriptor_points_regs(sp_descriptor,
#                                                         regions_id,
#                                                         elements_i)
#        _data = np.array(regions_id)
#        _data = _data.reshape((_data.shape[0], 1))
#    else:
#        regions, elements_i = get_regions4distances(sp_descriptor,
#                                                    elements, activated)
#        _data = np.array(regions)
#        _data = _data.reshape((_data.shape[0], 1))
#
#    ## 1. Computation of relations
#    relations = sp_descriptor.compute_net()[:, :, 0]
