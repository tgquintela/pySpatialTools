
"""
Util classes
------------
Classes which represent data types useful for the package pySpatialTools.

"""

## Selectors
#from spdesc_mapper import Sp_DescriptorMapper
from spdesc_mapper import DummySelector, GeneralCollectionSelectors,\
    Feat_RetrieverSelector, Spatial_RetrieverSelector,\
    FeatInd_RetrieverSelector

## Spatial elements collectors
from spatialelements import SpatialElementsCollection, Locations

## Membership relations
from Membership import Membership

## Mapper vals_i
from mapper_vals_i import Map_Vals_i, create_mapper_vals_i

## Neighbourhood information
from neighs_info import Neighs_Info
