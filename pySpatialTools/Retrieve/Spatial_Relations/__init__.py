
"""
Spatial relations
=================
Module which contains utilities for computing spatial relations between spatial
elements.
"""

## Class methods
from regionmetrics import CenterLocsRegionDistances,\
    ContiguityRegionDistances, PointsNeighsIntersection
from region_neighbourhood import OrderRegNeigh, SameRegNeigh,\
    LimDistanceRegNeigh


## Individual functions
from region_spatial_relations import regions_relation_points
from general_spatial_relations import general_spatial_relation,\
    general_spatial_relations
