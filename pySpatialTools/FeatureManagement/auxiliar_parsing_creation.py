
"""
Auxiliar parsing creation
-------------------------
Module which groups utils to instantiate classes easily to use in other parts
of the code.

"""

from pySpatialTools.Retrieve import _retriever_parsing_creation
from pySpatialTools.Discretization import _discretization_parsing_creation
from features_retriever import _features_parsing_creation
from spatial_descriptormodels import SpatialDescriptorModel


def _spdesc_parsing_creation(retrievers_info, features_info, pars_spdesc={},
                             selectors=None):
    """Instantiation of spdesc object from retriever and features information.
    """
    retrievers = _retriever_parsing_creation(retrievers_info)
    featurers = _features_parsing_creation(features_info)
    pars_spdesc['mapselector_spdescriptor'] = selectors
    spdesc = SpatialDescriptorModel(retrievers, featurers, **pars_spdesc)
    return spdesc

#
#def _aggregation_features_parsing_creation(aggregation_info):
#    """Instantiation of spdesc object from aggregation information."""
#    if isinstance(aggregation_info, SpatialDescriptorModel):
#        pass
#    elif type(aggregation_info) == tuple:
#        assert(len(aggregation_info) == 4)
#        disc_info, ret_in_info, ret_out_info, agg_info = aggregation_info
#        locs, regs, disc = _discretization_parsing_creation(disc_info)
#        ret_in_info = agg_info[0](locs, regs, disc, ret_in_info)
#
#        if len(aggregation_info) == 2:
#            retrievers_info, features_info = aggregation_info
#            aggregation_info =\
#                _spdesc_parsing_creation(retrievers_info, features_info)
#
#    return aggregation_info
