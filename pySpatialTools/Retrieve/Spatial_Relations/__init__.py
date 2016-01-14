
"""
Spatial relations
=================
Module which contains utilities for computing spatial relations between spatial
elements.
"""

## Class methods
from regionmetrics import CenterLocsRegionDistances,\
    ContiguityRegionDistances, PointsNeighsIntersection

## Individual functions for preparing the computation of relations
from aux_regionmetrics import get_regions4distances,\
    create_sp_descriptor_regionlocs, create_sp_descriptor_points_regs

## Individual functions for computing relations
from region_spatial_relations import regions_relation_points
from general_spatial_relations import general_spatial_relation,\
    general_spatial_relations
