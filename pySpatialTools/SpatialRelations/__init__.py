
"""
Spatial relations
=================
Module which contains utilities for computing spatial relations between spatial
elements and store them if it is needed.
It could be useful for:
- the representation of relations between explicit defined relations
    (e.g. set disc)
- the computation of distances between collections of elements (sets or
    regions), using the spatial information of its elements if it is available
    or the intersection betweeen elements.
- the transformation of relative posititions.
- the recomputation from a whole relative positions (from net).

"""

## Class containers of distances
from regionmetrics import DummyRegDistance, RegionDistances

## Function methods of distances
#from regiondistances_computers import compute_CenterLocsRegionDistances,\
#    compute_ContiguityRegionDistances, compute_PointsNeighsIntersection,\
#    compute_AvgDistanceRegions

## Individual functions for computing relations
#from region_spatial_relations import regions_relation_points
#from general_spatial_relations import general_spatial_relation,\
#    general_spatial_relations
from formatters import format_out_relations

## Individual functions for preparing the computation of relations
#from aux_regionmetrics import get_regions4distances,\
#    create_sp_descriptor_regionlocs, create_sp_descriptor_points_regs
