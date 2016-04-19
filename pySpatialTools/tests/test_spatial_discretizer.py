
"""
test spatial discretization
---------------------------
test for spatial discretizors.

"""

## Imports
import numpy as np
from itertools import product
#import matplotlib.pyplot as plt
from pySpatialTools.Discretization import *
from pySpatialTools.utils.artificial_data import *
from pySpatialTools.Discretization.Discretization_2d.utils import *

## TODO:
########
# IrregularSpatialDisc
# sklearnlike


def test():
    ## 0. Artificial data
    # Parameters
    n = 1000
    ngx, ngy = 100, 100
    # Artificial distribution in space
    fucnts = [lambda locs: locs[:, 0]*np.cos(locs[:, 1]*2*np.pi),
              lambda locs: locs[:, 0]*np.sin(locs[:, 1]*np.pi*2)]
    locs1 = random_transformed_space_points(n, 2, None)
    locs2 = random_transformed_space_points(n, 2, fucnts)
    locs3 = random_transformed_space_points(n/100, 2, None)
    Locs = [locs1, locs2, locs3]
    # Set discretization
    elements1 = np.random.randint(0, 50, 200)
    elements2 = np.random.randint(0, 2000, 200)
#
#    ## Test distributions
#    #fig1 = plt.plot(locs[:,0], locs[:, 1], '.')
#    #fig2 = plt.plot(locs2[:,0], locs2[:, 1], '.')
#
    ### Testing utils
    # Discretization 2d utils
    regions_id = np.arange(10)
    indices = np.unique(np.random.randint(0, 10, 5))
    indices_assignation(indices, regions_id)
    mask_application_grid(np.random.random(), np.random.random(10))
    #compute_limital_polygon(limits)
    #match_regions(polygons, regionlocs, n_dim=2)
    # testing voronoi tesselation
    tesselation(np.random.random((10, 2)))

    ### Discretization
    ############################# Grid discretizer ############################
    disc1 = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))
    disc2 = GridSpatialDisc((ngx, ngy), xlim=(-1, 1), ylim=(-1, 1))
    # Test functions
    disc1[0]
    disc1[locs1[0]]
    len(disc1)
    for i in range(len(Locs)):
        disc1.discretize(Locs[i])
        disc1.map_locs2regionlocs(Locs[i])
        disc1.map2agglocs(Locs[i])
        disc1._map_regionid2regionlocs(0)
        disc1._map_locs2regionlocs(Locs[i])
        disc1.retrieve_region(Locs[i][0], {})
        disc1.retrieve_neigh(Locs[i][0], Locs[i])
        disc1.get_activated_regions(Locs[i])
        disc1.belong_region(Locs[i])
        disc1.belong_region(Locs[i], disc1[0])
        disc1.check_neighbors([disc1[0], disc1[1]], disc1[2])

    ########################### Bisector discretizer ##########################
    try:
        boolean = False
        disc6 = BisectorSpatialDisc(locs3, np.arange(len(locs3)+1))
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    disc6 = BisectorSpatialDisc(locs3, np.arange(len(locs3)))
    # Test functions
    disc6[0]
    disc6[locs1[0]]
    len(disc6)
    for i in range(len(Locs)):
        disc6.discretize(Locs[i])
        disc6.map_locs2regionlocs(Locs[i])
        disc6.map2agglocs(Locs[i])
        disc6._map_regionid2regionlocs(0)
        disc6._map_locs2regionlocs(Locs[i])
        disc6.retrieve_region(Locs[i][0], {})
        disc6.retrieve_neigh(Locs[i][0], Locs[i])
        disc6.get_activated_regions(Locs[i])
        disc6.belong_region(Locs[i])
        disc6.belong_region(Locs[i], disc6[0])
        disc6.check_neighbors([disc6[0], disc6[1]], disc6[2])
        ## Not implemented yet
        #disc6.get_contiguity()
        #disc6.get_contiguity(disc6[0])
        #disc6.get_limits()
        #disc6.get_limits(disc6[0])

    ########################### Circular discretizer ##########################
    centers = np.random.random((20, 2))
    radios = np.random.random(20)/5
    jit = np.random.random((100, 2))
    locs_r = centers[np.random.randint(0, 20, 100)] + jit/100000.
    regions_ids = [np.arange(20), np.arange(10, 30)]

    for j in range(len(regions_ids)):
        disc4 = CircularInclusiveSpatialDisc(centers, radios, regions_ids[j])
        disc5 = CircularExcludingSpatialDisc(centers, radios, regions_ids[j])

        # Testing functions
        for i in range(len(Locs)):
            disc4.discretize(Locs[i])
#            disc4.map_locs2regionlocs(Locs[i])
#            disc4.map2agglocs(Locs[i])
#            disc4._map_regionid2regionlocs(0)
#            disc4._map_locs2regionlocs(Locs[i])
#            disc4.retrieve_region(Locs[i][0], {})
#            disc4.retrieve_neigh(Locs[i][0], Locs[i])
#            disc4.get_activated_regions(Locs[i])
#            disc4.belong_region(Locs[i])
#            disc4.belong_region(Locs[i], disc4[0])
#            disc4.check_neighbors([disc4[0], disc4[1]], disc4[2])
#            ## Not implemented yet
#            #disc4.get_contiguity()
#            #disc4.get_contiguity(disc6[0])
#            #disc4.get_limits(Locs[i])
#            #disc4.get_limits(Locs[i], disc6[0])
            disc5.discretize(Locs[i])
#            #disc5.map_locs2regionlocs(Locs[i])
#            disc5.map2agglocs(Locs[i])
#            disc5.retrieve_region(Locs[i][0], {})
#            disc5.retrieve_neigh(Locs[i][0], Locs[i])
#            disc5.get_activated_regions(Locs[i])
#            disc5.belong_region(Locs[i])
#            disc5.belong_region(Locs[i], disc5[0])
#            disc5.check_neighbors([disc5[0], disc5[1]], disc5[2])
#            ## Not implemented yet
#            #disc5.get_contiguity()
#            #disc5.get_contiguity(disc5[0])
#            #disc5.get_limits(Locs[i])
#            #disc5.get_limits(Locs[i], disc5[0])
        disc4.discretize(locs_r)
        #disc4.map_locs2regionlocs(locs_r)
        disc4.map2agglocs(locs_r)
        disc4.get_activated_regions(locs_r)
        disc4.belong_region(locs_r)
        disc4.belong_region(locs_r, disc4[0])

        disc5.discretize(locs_r)
        #disc5.map_locs2regionlocs(locs_r)
        disc5.map2agglocs(locs_r)
        disc5.get_activated_regions(locs_r)
        disc5.belong_region(locs_r)
        disc5.belong_region(locs_r, disc5[0])

    ############################## Set discretizer ############################
    disc6 = SetDiscretization(np.random.randint(0, 2000, 50))
    # Test functions
    disc6[0]
    disc6[disc6.regionlocs[0]]
    len(disc6)
    disc6.discretize(disc6.regionlocs)
    disc6._map_regionid2regionlocs(0)
    disc6.retrieve_region(disc6.regionlocs[0], {})
    disc6.retrieve_neigh(disc6.regionlocs[0], disc6.regionlocs)
    disc6.get_activated_regions(disc6.regionlocs)
    disc6.belong_region(disc6.regionlocs)
    disc6.belong_region(disc6.regionlocs, disc6[0])
    disc6.check_neighbors([disc6[0], disc6[1]], disc6[2])

    disc7 = SetDiscretization(randint_sparse_matrix(0.2, (2000, 100), 1))
    # Test functions
    disc7[0]
    disc7[disc7.regionlocs[0]]
    len(disc7)
    disc7.discretize(disc7.regionlocs)
    disc7._map_regionid2regionlocs(0)
    disc7.retrieve_region(disc7.regionlocs[0], {})
    disc7.retrieve_neigh(disc7.regionlocs[0], disc7.regionlocs)
    disc7.get_activated_regions(disc7.regionlocs)
#    disc7.belong_region(disc6.regionlocs)
#    disc7.belong_region(disc6.regionlocs, disc7[0])
#    disc7.check_neighbors([disc7[0], disc7[1]], disc7[2])

#
#    # Discretization action
#    regions = disc1.discretize(locs1)
#    regions = disc1.discretize(locs2)
#    regions = disc2.discretize(locs1)
#    regions = disc2.discretize(locs2)
#    regions = disc6.discretize(locs1)
#    regions = disc6.discretize(locs2)
#    regions = disc4.discretize(locs1)
#    regions = disc4.discretize(locs2)
#    regions = disc5.discretize(locs1)
#    regions = disc5.discretize(locs2)
#    regions = disc6.discretize(elements1)
#    regions7 = disc7.discretize(elements2)
#
#    a = randint_sparse_matrix(0.2, (2000, 100), 1)
#
#    # Inverse discretization action
#
#    # Contiguity
#    contiguity = disc1.get_contiguity()
#    contiguity = disc2.get_contiguity()
#    #contiguity = disc6.get_contiguity()
#    #contiguity = disc4.get_contiguity()
#    #contiguity = disc5.get_contiguity()
#    #contiguity = disc6.get_contiguity()
#    #contiguity = disc7.get_contiguity()
#
#    ## Other parameters and functions
#    disc1.borders, disc5.borders, disc6.borders
#
#    ## Extending coverage
#    n_in, n_out = 100, 20
#    relations = [np.unique(np.random.randint(0, n_out,
#                                             np.random.randint(n_out)))
#                 for i in range(n_in)]
#
#    disc8 = SetDiscretization(relations)
#    relations = [list(e) for e in relations]
#    disc9 = SetDiscretization(relations)
#    disc8.discretize(np.random.randint(0, 20, 100))
#    disc9.discretize(np.random.randint(0, 20, 100))
#
#    ## Check regions
#    neighs = disc1.check_neighbors(np.array([0, 1]), 0)
#
#    activated = disc1.get_activated_regions(locs1[:10])
#    activated = disc2.get_activated_regions(locs1[:10])
#    activated = disc6.get_activated_regions(locs1[:10])
#    activated = disc4.get_activated_regions(locs1[:10])
#    activated = disc5.get_activated_regions(locs1[:10])
#    activated = disc6.get_activated_regions(elements1[:10])
#    activated = disc7.get_activated_regions(elements2[:10])
#    activated = disc8.get_activated_regions(np.array([0, 1]))
#    activated = disc9.get_activated_regions(np.array([0, 1]))
#
#    disc1.belong_region(locs1[0], 0)
#    disc2.belong_region(locs1[1], 0)
#    limits = disc1.get_limits()
#    limits = disc1.get_limits(0)
#    limits = disc2.get_limits()
##    limits = disc6.get_limits()
##    limits = disc4.get_limits(0)
#
#    disc1._map_regionid2regionlocs(0)
#    disc2._map_regionid2regionlocs(0)
#    disc6._map_regionid2regionlocs(0)
#    disc4._map_regionid2regionlocs(0)
#    disc5._map_regionid2regionlocs(0)
#    disc6._map_regionid2regionlocs(0)
#    disc7._map_regionid2regionlocs(0)
#    disc8._map_regionid2regionlocs(0)
#    disc9._map_regionid2regionlocs(0)
