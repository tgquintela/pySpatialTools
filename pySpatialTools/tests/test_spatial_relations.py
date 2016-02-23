
"""
test SpatialRelations
---------------------
test for spatial relations module. There are two computers methods untested.

"""

import numpy as np
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import OrderEleNeigh

from pySpatialTools.SpatialRelations import RegionDistances, DummyRegDistance
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
    regs = np.unique(np.random.randint(0, 1000, 200))
    dummymapper = DummyRegDistance(regs)

    ## Test functions
    mainmapper1[0]
    mainmapper1[mainmapper1.data[0]]
    mainmapper1.retrieve_neighs(0)
    mainmapper1.retrieve_neighs(mainmapper1.data[0])
    mainmapper1.data
    mainmapper1.data_input
    mainmapper1.data_output
    mainmapper1.shape

    mainmapper2[0]
    mainmapper2[mainmapper2.data[0]]
    mainmapper2.retrieve_neighs(0)
    mainmapper2.retrieve_neighs(mainmapper2.data[0])
    mainmapper2.data
    mainmapper2.data_input
    mainmapper2.data_output
    mainmapper2.shape

    dummymapper[0]
    dummymapper[dummymapper.data[0]]
    dummymapper.retrieve_neighs(0)
    dummymapper.retrieve_neighs(dummymapper.data[0])
    dummymapper.data
    dummymapper.data_input
    dummymapper.data_output
    dummymapper.shape

    ## In retrievers
    ret0 = OrderEleNeigh(dummymapper, info_ret)
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
