
"""
test SpatialRelations
---------------------
test for spatial relations module. There are two computers methods untested.

"""

import numpy as np
from itertools import product
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import OrderEleNeigh, KRetriever
from pySpatialTools.utils.artificial_data import randint_sparse_matrix
from pySpatialTools.FeatureManagement.Descriptors import DummyDescriptor

from pySpatialTools.SpatialRelations import RegionDistances, DummyRegDistance,\
    format_out_relations, _relations_parsing_creation
from pySpatialTools.SpatialRelations.regiondistances_computers\
    import compute_AvgDistanceRegions, compute_CenterLocsRegionDistances,\
    compute_ContiguityRegionDistances  # , compute_PointsNeighsIntersection
from pySpatialTools.SpatialRelations.util_spatial_relations import\
    general_spatial_relation, general_spatial_relations
from pySpatialTools.SpatialRelations.element_metrics import\
    measure_difference, unidimensional_periodic
from pySpatialTools.SpatialRelations.aux_regionmetrics import\
    get_regions4distances, filter_possible_neighs

from pySpatialTools.SpatialRelations import\
    format_out_relations, get_regions4distances,\
    sparse_from_listaregneighs, compute_selfdistances

from pySpatialTools.SpatialRelations.relative_positioner import\
    RelativePositioner, RelativeRegionPositioner, metric_distances,\
    diff_vectors


def test():
    ### Parameters or externals
    info_ret = {'order': 2}
    locs = np.random.random((1000, 2))
    inttypes = [int, np.int32, np.int64]
#    mainmapper1 = generate_random_relations(25, store='sparse')
#    mainmapper2 = generate_random_relations(100, store='sparse')
#    mainmapper3 = generate_random_relations(5000, store='sparse')
    griddisc1 = GridSpatialDisc((5, 5), xlim=(0, 1), ylim=(0, 1))
    griddisc2 = GridSpatialDisc((10, 10), xlim=(0, 1), ylim=(0, 1))
    griddisc3 = GridSpatialDisc((50, 100), xlim=(0, 1), ylim=(0, 1))

    ### Testing utities
    ## util_spatial_relations
    f = lambda x, y: x + y
    sp_elements = np.random.random(10)
    general_spatial_relation(sp_elements[0], sp_elements[0], f)
    general_spatial_relations(sp_elements, f, simmetry=False)
    general_spatial_relations(sp_elements, f, simmetry=True)

    ## format_out_relations
    mainmapper1 = randint_sparse_matrix(0.8, (25, 25))
    format_out_relations(mainmapper1, 'sparse')
    format_out_relations(mainmapper1, 'network')
    format_out_relations(mainmapper1, 'sp_relations')
    lista = format_out_relations(mainmapper1, 'list')
    u_regs = mainmapper1.data

    ## Element metrics
    element_i, element_j = 54, 2
    pars1 = {'periodic': 60}
    pars2 = {}
    unidimensional_periodic(element_i, element_j, pars=pars1)
    unidimensional_periodic(element_i, element_j, pars=pars2)
    unidimensional_periodic(element_j, element_i, pars=pars1)
    unidimensional_periodic(element_j, element_i, pars=pars2)
    measure_difference(element_i, element_j, pars=pars1)
    measure_difference(element_i, element_j, pars=pars2)
    measure_difference(element_j, element_i, pars=pars1)
    measure_difference(element_j, element_i, pars=pars2)

    def ensure_output(neighs, dists, mainmapper):
#        print dists
        assert(all([len(e.shape) == 2 for e in dists]))
        assert(all([len(e) == 0 for e in dists if np.prod(e.shape) == 0]))
        if mainmapper._out == 'indices':
#            print neighs
            correcness = []
            for nei in neighs:
                if len(nei):
                    correcness.append(all([type(e) in inttypes for e in nei]))
                else:
                    correcness.append(nei.dtype in inttypes)
            assert(correcness)

    ###########################################################################
    ### Massive combinatorial testing
    # Possible parameters
    relations, pars_rel, _data =\
        compute_ContiguityRegionDistances(griddisc1, store='sparse')
    pos_relations = [relations, relations.A,
                     format_out_relations(relations, 'network')]
    pos_distanceorweighs, pos_sym = [True, False], [True, False]
    pos_inputstypes, pos_outputtypes = [[None, 'indices', 'elements_id']]*2
    pos_input_type = [None, 'general', 'integer', 'array', 'array1', 'array2',
                      'list', 'list_int', 'list_array']
    pos_inputs = [[0], 0, 0, np.array([0]), np.array([0]), np.array([0]),
                  [0], [0], [np.array([0])]]
    pos_data_in, pos_data = [[]]*2
    possibles = [pos_relations, pos_distanceorweighs, pos_sym, pos_outputtypes,
                 pos_inputstypes, pos_input_type]
    # Combinations
    for p in product(*possibles):
        mainmapper1 = RegionDistances(relations=p[0], distanceorweighs=p[1],
                                      symmetric=p[2], output=p[3], input_=p[4],
                                      input_type=p[5])
        mainmapper1[slice(0, 1)]
        # Define input
        if p[5] is None:
            if p[4] != 'indices':
                neighs, dists = mainmapper1[mainmapper1.data[0]]
                ensure_output(neighs, dists, mainmapper1)
                neighs, dists = mainmapper1[0]
                ensure_output(neighs, dists, mainmapper1)
                neighs, dists = mainmapper1[np.array([-1])]
                ensure_output(neighs, dists, mainmapper1)
            if p[4] != 'elements_id':
                neighs, dists = mainmapper1[0]
                ensure_output(neighs, dists, mainmapper1)
                try:
                    boolean = False
                    mainmapper1[-1]
                    boolean = True
                    raise Exception("It has to halt here.")
                except:
                    if boolean:
                        raise Exception("It has to halt here.")
        else:
            if p[5] == 'list':
                # Get item
                neighs, dists = mainmapper1[[0]]
                ensure_output(neighs, dists, mainmapper1)
                neighs, dists = mainmapper1[[np.array([0])]]
                ensure_output(neighs, dists, mainmapper1)
                try:
                    boolean = False
                    mainmapper1[[None]]
                    boolean = True
                    raise Exception("It has to halt here.")
                except:
                    if boolean:
                        raise Exception("It has to halt here.")
            idxs = pos_inputs[pos_input_type.index(p[5])]
            neighs, dists = mainmapper1[idxs]
            ensure_output(neighs, dists, mainmapper1)
        # Functions
        mainmapper1.set_inout(p[5], p[4], p[3])
        mainmapper1.transform(lambda x: x)
        mainmapper1.data
        mainmapper1.data_input
        mainmapper1.data_output
        mainmapper1.shape
        ## Extreme cases

    ## Individual extreme cases
    ## Instantiation
    pars_rel = {'symmetric': False}
    relations = relations.A
    data_in = list(np.arange(len(relations)))
    wrond_data = np.random.random((100))
    mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
                                  data_in=data_in, **pars_rel)
    data_in = np.arange(len(relations)).reshape((len(relations), 1))
    mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
                                  data_in=data_in, **pars_rel)
    try:
        boolean = False
        wrond_data = np.random.random((100, 3, 4))
        mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
                                      data_in=data_in, **pars_rel)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        wrond_data = np.random.random((100, 3, 4))
        sparse_rels = randint_sparse_matrix(0.8, (25, 25))
        mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
                                      data_in=data_in, **pars_rel)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        wrond_data = np.random.random((100, 3, 4))
        mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
                                      data_in=data_in, **pars_rel)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    relations = np.random.random((20, 20))
    try:
        boolean = False
        wrond_data = np.random.random((100, 3, 4))
        mainmapper3 = RegionDistances(relations=relations, _data=wrond_data,
                                      data_in=data_in, **pars_rel)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ## Other cases
    # Dummymap instantiation
    regs0 = np.unique(np.random.randint(0, 1000, 200))
    regs1 = regs0.reshape((len(regs0), 1))
    regs2 = regs0.reshape((len(regs0), 1, 1))
    pos_regs = [regs0, list(regs0), regs1]
    possibles = [pos_regs, pos_input_type]
    for p in product(*possibles):
        dummymapper = DummyRegDistance(p[0], p[1])
        # Get item
        idxs = pos_inputs[pos_input_type.index(p[1])]
        neighs, dists = dummymapper[idxs]
        ensure_output(neighs, dists, dummymapper)
        neighs, dists = dummymapper[slice(0, 1)]
        ensure_output(neighs, dists, dummymapper)
        ## Functions
        dummymapper.transform(lambda x: x)
        dummymapper.data
        dummymapper.data_input
        dummymapper.data_output
        dummymapper.shape
    # Halting cases
    try:
        boolean = False
        dummymapper = DummyRegDistance(regs2)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        dummymapper = DummyRegDistance(None)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        dummymapper[None]
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        dummymapper[-1]
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ###########################################################################
    ### Auxiliar parsing creation functions test
    ############################################
    # Standarts
    #    * relations object
    #    * (main_relations_info, pars_rel)
    #    * (main_relations_info, pars_rel, _data)
    #    * (main_relations_info, pars_rel, _data, data_in)
    #
    ## Main relations information
    relations = np.random.random((100, 20))
    _data = np.arange(20)
    _data_input = np.arange(100)

    relations_info = (relations, {})
    relations_object = _relations_parsing_creation(relations_info)
    assert(isinstance(relations_object, RegionDistances))

    relations_info = (relations, {}, _data)
    relations_object = _relations_parsing_creation(relations_info)
    assert(isinstance(relations_object, RegionDistances))

    relations_info = (relations, {}, _data, _data_input)
    relations_object = _relations_parsing_creation(relations_info)
    assert(isinstance(relations_object, RegionDistances))

    relations_object = _relations_parsing_creation(relations_object)
    assert(isinstance(relations_object, RegionDistances))

    relations_object = _relations_parsing_creation(relations)
    assert(isinstance(relations_object, RegionDistances))

    ###########################################################################
    ### Computers testing
    #####################
    ## aux_regionmetrics
    # Get regions activated
    elements = griddisc1.get_regions_id()
    get_regions4distances(griddisc1, elements=None, activated=None)
    get_regions4distances(griddisc1, elements, activated=elements)

    # Filter possible neighs
    only_possible = np.unique(np.random.randint(0, 100, 50))
    neighs = [np.unique(np.random.randint(0, 100, 6)) for i in range(4)]
    dists = [np.random.random(len(neighs[i])) for i in range(4)]
    filter_possible_neighs(only_possible, neighs, dists)
    filter_possible_neighs(only_possible, neighs, None)

    # TODO: Sync with other classes as sp_desc_models
    lista = [[0, 1, 2, 3], [0, 2, 3, 5], [1, 1, 1, 1]]
    u_regs = np.arange(25)
    regions_id = np.arange(25)
    elements_i = np.arange(25)
    element_labels = np.arange(25)
    discretizor = GridSpatialDisc((5, 5), xlim=(0, 1), ylim=(0, 1))

    locs = np.random.random((100, 2))
    retriever = KRetriever
    info_ret = np.ones(100)*4
    descriptormodel = DummyDescriptor()

    sp_descriptor = discretizor, locs, retriever, info_ret, descriptormodel

    sparse_from_listaregneighs(lista, u_regs, symmetric=True)
    sparse_from_listaregneighs(lista, u_regs, symmetric=False)
    ret_selfdists = KRetriever(locs, 4, ifdistance=True)
    compute_selfdistances(ret_selfdists, np.arange(100), typeoutput='network',
                          symmetric=True)
    compute_selfdistances(ret_selfdists, np.arange(100), typeoutput='sparse',
                          symmetric=True)
    compute_selfdistances(ret_selfdists, np.arange(100), typeoutput='matrix',
                          symmetric=True)

#    create_sp_descriptor_points_regs(sp_descriptor, regions_id, elements_i)
#    create_sp_descriptor_regionlocs(sp_descriptor, regions_id, elements_i)

    ## Compute Avg distance
    locs = np.random.random((100, 2))
    sp_descriptor = (griddisc1, locs), (KRetriever, {'info_ret': 5}), None
    relations, pars_rel, _data =\
        compute_AvgDistanceRegions(sp_descriptor, store='network')
    regdists = RegionDistances(relations=relations, _data=_data, **pars_rel)
    relations, pars_rel, _data =\
        compute_AvgDistanceRegions(sp_descriptor, store='matrix')
    regdists = RegionDistances(relations=relations, _data=_data, **pars_rel)
    relations, pars_rel, _data =\
        compute_AvgDistanceRegions(sp_descriptor, store='sparse')
    regdists = RegionDistances(relations=relations, _data=_data, **pars_rel)

    # Region spatial relations
    # For future (TODO)

    ### RegionDistances Computers
    griddisc1 = GridSpatialDisc((5, 5), xlim=(0, 1), ylim=(0, 1))
    ## Compute Contiguity
    relations, pars_rel, _data =\
        compute_ContiguityRegionDistances(griddisc1, store='matrix')
    relations, pars_rel, _data =\
        compute_ContiguityRegionDistances(griddisc1, store='network')
    relations, pars_rel, _data =\
        compute_ContiguityRegionDistances(griddisc1, store='sparse')
    mainmapper1 = RegionDistances(relations=relations, _data=None,
                                  **pars_rel)
    neighs, dists = mainmapper1.retrieve_neighs([0])
    assert(len(neighs) == len(dists))
    assert(len(neighs) == 1)
    neighs, dists = mainmapper1.retrieve_neighs([0, 1])
    assert(len(neighs) == len(dists))
    assert(len(neighs) == 2)

    ## Compute CenterLocs
    sp_descriptor = griddisc1, None, None
    ## TODO: pdist problem
    relations, pars_rel, _data =\
        compute_CenterLocsRegionDistances(sp_descriptor, store='network',
                                          elements=None, symmetric=True,
                                          activated=None)
    relations, pars_rel, _data =\
        compute_CenterLocsRegionDistances(sp_descriptor, store='matrix',
                                          elements=None, symmetric=True,
                                          activated=None)
    relations, pars_rel, _data =\
        compute_CenterLocsRegionDistances(sp_descriptor, store='sparse',
                                          elements=None, symmetric=True,
                                          activated=None)
    sp_descriptor = griddisc1, (KRetriever, {'info_ret': 2}), None

### TODO: Descriptormodel
    relations, pars_rel, _data =\
        compute_CenterLocsRegionDistances(sp_descriptor, store='sparse',
                                          elements=None, symmetric=True,
                                          activated=None)

    ## Retriever tuple
    ## Retriever object

    ## Spdesc

    ## Compute PointsNeighsIntersection

    ## Aux_regionmetrics
    #sparse_from_listaregneighs(lista, u_regs, symmetric)

    ###########################################################################
    ### Relative positioner testing
    ###############################
    n_el, n_dim = 5, 2
    elements_i = np.random.random((n_el, n_dim))
    elements_neighs = []
    for i in range(n_el):
        aux_neigh = np.random.random((np.random.randint(1, 4), n_dim))
        elements_neighs.append(aux_neigh)
    rel_pos = RelativePositioner(metric_distances)
    rel_pos.compute(elements_i, elements_neighs)

    rel_pos = RelativePositioner(diff_vectors)
    rel_pos.compute(elements_i, elements_neighs)

#    RelativeRegionPositioner
