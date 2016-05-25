
"""
Auxiliar parsing creation
-------------------------
Module which groups utils to instantiate classes easily to use in other parts
of the code.

"""

from pySpatialTools.Retrieve import _retriever_parsing_creation
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
