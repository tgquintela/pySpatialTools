
"""
test SpatialRelations
---------------------
test for spatial relations module. There are two computers methods untested.

"""

import numpy as np
from itertools import product
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import OrderEleNeigh
from pySpatialTools.utils.artificial_data import randint_sparse_matrix

from pySpatialTools.SpatialRelations import RegionDistances, DummyRegDistance,\
    format_out_relations
from pySpatialTools.SpatialRelations.regiondistances_computers\
    import compute_AvgDistanceRegions, compute_CenterLocsRegionDistances,\
    compute_ContiguityRegionDistances, compute_PointsNeighsIntersection
from pySpatialTools.SpatialRelations.util_spatial_relations import\
    general_spatial_relation, general_spatial_relations
from pySpatialTools.SpatialRelations.element_metrics import\
    measure_difference, unidimensional_periodic
from pySpatialTools.SpatialRelations.aux_regionmetrics import\
    get_regions4distances, filter_possible_neighs


#def test():
#    ### Parameters or externals
#    info_ret = {'order': 2}
#    locs = np.random.random((1000, 2))
##    mainmapper1 = generate_random_relations(25, store='sparse')
##    mainmapper2 = generate_random_relations(100, store='sparse')
##    mainmapper3 = generate_random_relations(5000, store='sparse')
#    griddisc1 = GridSpatialDisc((5, 5), xlim=(0, 1), ylim=(0, 1))
#    griddisc2 = GridSpatialDisc((10, 10), xlim=(0, 1), ylim=(0, 1))
#    griddisc3 = GridSpatialDisc((50, 100), xlim=(0, 1), ylim=(0, 1))
#
#    ### Testing utities
#    ## util_spatial_relations
#    f = lambda x, y: x + y
#    sp_elements = np.random.random(10)
#    general_spatial_relation(sp_elements[0], sp_elements[0], f)
#    general_spatial_relations(sp_elements, f, simmetry=False)
#    general_spatial_relations(sp_elements, f, simmetry=True)
#
#    ## format_out_relations
#    mainmapper1 = randint_sparse_matrix(0.8, (25, 25))
#    format_out_relations(mainmapper1, 'sparse')
#    format_out_relations(mainmapper1, 'network')
#    format_out_relations(mainmapper1, 'sp_relations')
#    lista = format_out_relations(mainmapper1, 'list')
#
#    ## Element metrics
#    element_i, element_j = 54, 2
#    pars1 = {'periodic': 60}
#    pars2 = {}
#    unidimensional_periodic(element_i, element_j, pars=pars1)
#    unidimensional_periodic(element_i, element_j, pars=pars2)
#    unidimensional_periodic(element_j, element_i, pars=pars1)
#    unidimensional_periodic(element_j, element_i, pars=pars2)
#    measure_difference(element_i, element_j, pars=pars1)
#    measure_difference(element_i, element_j, pars=pars2)
#    measure_difference(element_j, element_i, pars=pars1)
#    measure_difference(element_j, element_i, pars=pars2)
#    ## Relative position
#
#    ## aux_regionmetrics
#    # Get regions activated
#    elements = griddisc1.get_regions_id()
#    get_regions4distances(griddisc1, elements=None, activated=None)
#    get_regions4distances(griddisc1, elements, activated=elements)
#
#    # Filter possible neighs
#    only_possible = np.unique(np.random.randint(0, 100, 50))
#    neighs = [np.unique(np.random.randint(0, 100, 6)) for i in range(4)]
#    dists = [np.random.random(len(neighs[i])) for i in range(4)]
#    filter_possible_neighs(only_possible, neighs, dists)
#    filter_possible_neighs(only_possible, neighs, None)
#
#    # TODO: Sync with other classes as sp_desc_models
##    sparse_from_listaregneighs(lista, u_regs, symmetric=True)
##    sparse_from_listaregneighs(lista, u_regs, symmetric=False)
##    create_sp_descriptor_points_regs(sp_descriptor, regions_id, elements_i)
##    create_sp_descriptor_regionlocs(sp_descriptor, regions_id, elements_i)
##    compute_selfdistances(retriever, element_labels, typeoutput='network',
##                          symmetric=True)
#    # Region spatial relations
#    # For future (TODO)
#
#    ### RegionDistances Computers
#    ## Compute Contiguity
#    relations, _data, symmetric, store =\
#        compute_ContiguityRegionDistances(griddisc1, store='sparse')
#    mainmapper1 = RegionDistances(relations=relations, _data=None,
#                                  symmetric=symmetric)
#    neighs, dists = mainmapper1.retrieve_neighs([0])
#    assert(len(neighs) == len(dists))
#    assert(len(neighs) == 1)
#    neighs, dists = mainmapper1.retrieve_neighs([0, 1])
#    assert(len(neighs) == len(dists))
#    assert(len(neighs) == 2)
#
#    def ensure_output(neighs, dists):
#        pass
#
#    ###########################################################################
#    ### Massive combinatorial testing
#    # Possible parameters
#    pos_relations = [relations, relations.A,
#                     format_out_relations(relations, 'network')]
#    pos_distanceorweighs, pos_sym = [True, False], [True, False]
#    pos_inputstypes, pos_outputtypes = [[None, 'indices', 'elements_id']]*2
#    pos_input_type = [None, 'general', 'integer', 'array', 'array1', 'array2',
#                      'list', 'list_int', 'list_array']
#    pos_inputs = [[0], 0, 0, np.array([0]), np.array([0]), np.array([0]),
#                  [0], [0], [np.array([0])]]
#    pos_data_in, pos_data = [[]]*2
#    possibles = [pos_relations, pos_distanceorweighs, pos_sym, pos_outputtypes,
#                 pos_inputstypes, pos_input_type]
#    # Combinations
#    for p in product(*possibles):
#        mainmapper1 = RegionDistances(relations=p[0], distanceorweighs=p[1],
#                                      symmetric=p[2], output=p[3], input_=p[4],
#                                      input_type=p[5])
#        mainmapper1[slice(0, 1)]
#        # Define input
#        if p[5] is None:
#            if p[4] != 'indices':
#                neighs, dists = mainmapper1[mainmapper1.data[0]]
#                ensure_output(neighs, dists)
#                neighs, dists = mainmapper1[0]
#                ensure_output(neighs, dists)
#                neighs, dists = mainmapper1[np.array([-1])]
#                ensure_output(neighs, dists)
#            if p[4] != 'elements_id':
#                neighs, dists = mainmapper1[0]
#                ensure_output(neighs, dists)
#                try:
#                    boolean = False
#                    mainmapper1[-1]
#                    boolean = True
#                except:
#                    if boolean:
#                        raise Exception("It has to halt here.")
#        else:
#            if p[5] == 'list':
#                # Get item
#                neighs, dists = mainmapper1[[0]]
#                ensure_output(neighs, dists)
#                neighs, dists = mainmapper1[[np.array([0])]]
#                ensure_output(neighs, dists)
#                try:
#                    boolean = False
#                    mainmapper1[[None]]
#                    boolean = True
#                except:
#                    if boolean:
#                        raise Exception("It has to halt here.")
#            idxs = pos_inputs[pos_input_type.index(p[5])]
#            neighs, dists = mainmapper1[idxs]
#            ensure_output(neighs, dists)
#        # Functions
#        mainmapper1.set_inout(p[5], p[4], p[3])
#        mainmapper1.transform(lambda x: x)
#        mainmapper1.data
#        mainmapper1.data_input
#        mainmapper1.data_output
#        mainmapper1.shape
#        ## Extreme cases
#
#    ## Individual extreme cases
#    ## Instantiation
#    relations = relations.A
#    data_in = list(np.arange(len(relations)))
#    mainmapper3 = RegionDistances(relations=relations, _data=data_in,
#                                  symmetric=symmetric, data_in=data_in)
#    data_in = np.arange(len(relations)).reshape((len(relations), 1))
#    mainmapper3 = RegionDistances(relations=relations, _data=data_in,
#                                  symmetric=symmetric, data_in=data_in)
#    try:
#        boolean = False
#        wrond_data = np.random.random((100, 3, 4))
#        mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
#                                      symmetric=symmetric, data_in=data_in)
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        wrond_data = np.random.random((100, 3, 4))
#        sparse_rels = randint_sparse_matrix(0.8, (25, 25))
#        mainmapper3 = RegionDistances(relations=sparse_rels, _data=wrond_data,
#                                      symmetric=symmetric, data_in=data_in)
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        wrond_data = np.random.random((100, 3, 4))
#        mainmapper3 = RegionDistances(relations=relations, _data=data_in,
#                                      symmetric=symmetric, data_in=wrond_data)
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    relations = np.random.random((20, 20))
#    try:
#        boolean = False
#        wrond_data = np.random.random((100, 3, 4))
#        mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
#                                      symmetric=symmetric, data_in=data_in)
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#
#    ## Other cases
#    # Dummymap instantiation
#    regs0 = np.unique(np.random.randint(0, 1000, 200))
#    regs1 = regs0.reshape((len(regs0), 1))
#    regs2 = regs0.reshape((len(regs0), 1, 1))
#    pos_regs = [regs0, list(regs0), regs1]
#    possibles = [pos_regs, pos_input_type]
#    for p in product(*possibles):
#        dummymapper = DummyRegDistance(p[0], p[1])
#        # Get item
#        idxs = pos_inputs[pos_input_type.index(p[1])]
#        neighs, dists = dummymapper[idxs]
#        ensure_output(neighs, dists)
#        neighs, dists = dummymapper[slice(0, 1)]
#        ensure_output(neighs, dists)
#        ## Functions
#        dummymapper.transform(lambda x: x)
#        dummymapper.data
#        dummymapper.data_input
#        dummymapper.data_output
#        dummymapper.shape
#    # Halting cases
#    try:
#        boolean = False
#        dummymapper = DummyRegDistance(regs2)
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        dummymapper = DummyRegDistance(None)
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        dummymapper[None]
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        dummymapper[-1]
#        boolean = True
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#
#    ## Needed regionmetrics
##    mainmapper2 = RegionDistances(relations=relations, _data=_data,
##                                  symmetric=symmetric)
##    ## In retrievers
##    ret0 = OrderEleNeigh(dummymapper, info_ret, constant_info=True)
##    ret1 = OrderEleNeigh(mainmapper1, info_ret, bool_input_idx=False,
##                         constant_info=True)
##    ret2 = OrderEleNeigh(mainmapper2, info_ret, constant_info=True)
##    ret3 = OrderEleNeigh(mainmapper3, info_ret)
#
#    ## Compute Avg distance
##    relations, _data, symmetric, store =\
##        compute_AvgDistanceRegions(locs, griddisc1, ret1)
##    regdists = RegionDistances(relations=relations, _data=_data,
##                               symmetric=symmetric)
##    regdists = RegionDistances(relations=relations, _data=None,
##                               symmetric=symmetric, distanceorweighs=False)
##    relations, _data, symmetric, store =\
##        compute_AvgDistanceRegions(locs, griddisc2, ret2)
##    regdists = RegionDistances(relations=relations, _data=_data,
##                               symmetric=symmetric)
##    relations, _data, symmetric, store =\
##        compute_AvgDistanceRegions(locs, griddisc3, ret3)
##    regdists = RegionDistances(relations=relations, _data=_data,
##                               symmetric=symmetric)
#
#
#    ## Compute CenterLocs
#    ## Compute PointsNeighsIntersection
#
#
#    ## Aux_regionmetrics
#    #sparse_from_listaregneighs(lista, u_regs, symmetric)
