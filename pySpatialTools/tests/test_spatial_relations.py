
"""
test SpatialRelations
---------------------
test for spatial relations module. There are two computers methods untested.

"""

import numpy as np
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import OrderEleNeigh

from pySpatialTools.SpatialRelations import RegionDistances
from pySpatialTools.SpatialRelations.regiondistances_computers\
    import compute_AvgDistanceRegions, compute_CenterLocsRegionDistances,\
    compute_ContiguityRegionDistances, compute_PointsNeighsIntersection


def test():
    info_ret = {'order': 2}
    locs = np.random.random((10000, 2))

#    mainmapper1 = generate_random_relations(25, store='sparse')
#    mainmapper2 = generate_random_relations(100, store='sparse')
#    mainmapper3 = generate_random_relations(5000, store='sparse')
    griddisc1 = GridSpatialDisc((5, 5), xlim=(0, 1), ylim=(0, 1))
    griddisc2 = GridSpatialDisc((10, 10), xlim=(0, 1), ylim=(0, 1))
    griddisc3 = GridSpatialDisc((50, 100), xlim=(0, 1), ylim=(0, 1))

    ## Compute Contiguity
    relations, _data, symmetric, store =\
        compute_ContiguityRegionDistances(griddisc1, store='sparse')
    mainmapper1 = RegionDistances(relations=relations, _data=_data,
                                  symmetric=symmetric)
    relations, _data, symmetric, store =\
        compute_ContiguityRegionDistances(griddisc2, store='sparse')
    mainmapper2 = RegionDistances(relations=relations, _data=_data,
                                  symmetric=symmetric)
#    relations, _data, symmetric, store =\
#        compute_ContiguityRegionDistances(griddisc3, store='sparse')
#    mainmapper3 = RegionDistances(relations=relations, _data=_data,
#                                  symmetric=symmetric)

    ret1 = OrderEleNeigh(mainmapper1, info_ret)
    ret2 = OrderEleNeigh(mainmapper2, info_ret)
#    ret3 = OrderEleNeigh(mainmapper3, info_ret)

    ## Compute Avg distance
    relations, _data, symmetric, store =\
        compute_AvgDistanceRegions(locs, griddisc1, ret1)
    regdists = RegionDistances(relations=relations, _data=_data,
                               symmetric=symmetric)
    relations, _data, symmetric, store =\
        compute_AvgDistanceRegions(locs, griddisc2, ret2)
    regdists = RegionDistances(relations=relations, _data=_data,
                               symmetric=symmetric)
#    relations, _data, symmetric, store =\
#        compute_AvgDistanceRegions(locs, griddisc3, ret3)
#    regdists = RegionDistances(relations=relations, _data=_data,
#                               symmetric=symmetric)


    ## Compute CenterLocs
    ## Compute PointsNeighsIntersection
