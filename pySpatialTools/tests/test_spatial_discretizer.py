
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
from pySpatialTools.Discretization.utils import check_discretizors
from pySpatialTools.Discretization.Discretization_2d.circdiscretization import\
    CircularSpatialDisc
from pySpatialTools.Discretization.Discretization_2d.griddiscretization import\
    mapping2grid, compute_contiguity_grid
from pySpatialTools.Discretization.Discretization_2d.utils import *
from pySpatialTools.Discretization.Discretization_set.\
    general_set_discretization import format_membership, to_sparse,\
    find_idx_in_array

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

    # Checker
    try:
        boolean = False
        check_discretizors(5)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    class Prueba:
        def __init__(self):
            self.n_dim = 9
            self.metric = None
            self.format_ = None
    try:
        boolean = False
        check_discretizors(Prueba)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    class Prueba:
        def __init__(self):
            self._compute_limits = 9
            self._compute_contiguity_geom = None
            self._map_loc2regionid = None
            self._map_regionid2regionlocs = None
    try:
        boolean = False
        check_discretizors(Prueba)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ### Discretization
    ############################# Grid discretizer ############################
    disc1 = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))
    disc2 = GridSpatialDisc((ngx, ngy), xlim=(-1, 1), ylim=(-1, 1))

    mapping2grid(Locs[0], (ngx, ngy), xlim=(0, 1), ylim=(0, 1))
    compute_contiguity_grid(-10, (ngx, ngy))

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
        disc1.get_limits()
        disc1.get_limits(disc1[0])
        # General functions applyable to this class
        disc1.map_locs2regionlocs(Locs[i])
        disc1.map2agglocs(Locs[i])
        disc1.get_contiguity()

        ## Special functions (recomendable to elevate in the hierharchy)
        disc1.get_nregs()
        disc1.get_regions_id()
        disc1.get_regionslocs()
        ## Class functions
        disc1._compute_contiguity_geom()
        disc1._compute_contiguity_geom(disc1[0])

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
    # Parameters
    centers = np.random.random((20, 2))
    radios = np.random.random(20)/5
    jit = np.random.random((100, 2))
    locs_r = centers[np.random.randint(0, 20, 100)] + jit/100000.
    regions_ids = [np.arange(20), np.arange(10, 30)]

    # General tests
    disc4 = CircularSpatialDisc(centers, 0.5)
    disc4._compute_limits(disc4.regions_id[0])
    disc4._map_regionid2regionlocs(0)
    try:
        boolean = False
        disc4._map_regionid2regionlocs(-1)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    # Exhaustive tests
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
    ## Format auxiliar functions
    random_membership(10, 20, True)
    random_membership(10, 20, False)
    memb, out = format_membership(randint_sparse_matrix(0.2, (2000, 100), 1))
#    to_sparse(memb, out)
#    memb, out = format_membership(np.random.random((20, 10)))
    to_sparse(memb, out)
    memb, out = format_membership(list_membership(20, 10))
    to_sparse(memb, out)
    memb0 = list_membership(10, 20)
    memb0 = [list(e) for e in memb0]
    memb, out = format_membership(memb0)
    to_sparse(memb, out)
    memb0 = list_membership(10, 20)
    memb0[0] = 0
    memb, out = format_membership(memb0)
    to_sparse(memb, out)
    memb0 = [[np.random.randint(0, 20)] for e in range(10)]
    memb, out = format_membership(memb0)
    to_sparse(memb, out)
    memb0 = list_membership(10, 20)
    memb0 = [dict(zip(m, len(m)*[{}])) for m in memb0]
    memb, out = format_membership(memb0)
    to_sparse(memb, out)
    find_idx_in_array(10, np.arange(40))

    ## Format discretizer
    disc6 = SetDiscretization(np.random.randint(0, 2000, 50))
    # Test functions
    disc6[0]
    disc6[disc6.regionlocs[0]]
    len(disc6)
    disc6.borders
    disc6.discretize(disc6.regionlocs)
    disc6._map_regionid2regionlocs(0)
    disc6.retrieve_region(disc6.regionlocs[0], {})
    disc6.retrieve_neigh(disc6.regionlocs[0], disc6.regionlocs)
    disc6.get_activated_regions(disc6.regionlocs)
    disc6.belong_region(disc6.regionlocs)
    disc6.belong_region(disc6.regionlocs, disc6[0])
    disc6.check_neighbors([disc6[0], disc6[1]], disc6[2])
#    disc6.get_limits()
#    disc6.get_limits(disc6.regions_id[0])

    disc7 = SetDiscretization(randint_sparse_matrix(0.2, (2000, 100), 1))
    # Test functions
    disc7[0]
    disc7.borders
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
#    disc7.get_limits()
#    disc7.get_limits(disc6.regions_id[0])
