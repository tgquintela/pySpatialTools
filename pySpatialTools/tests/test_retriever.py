
"""
test retrievers
---------------
test for retrievers precoded and framework of retrievers.

"""

import numpy as np
from itertools import product

from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    RetrieverManager, SameEleNeigh, OrderEleNeigh, LimDistanceEleNeigh,\
    DummyRetriever, GeneralRetriever, WindowsRetriever
from pySpatialTools.Retrieve.aux_retriever import _check_retriever

## WindowsRetriever functions
from pySpatialTools.Retrieve.aux_windowretriever import create_window_utils,\
    windows_iteration, create_map2indices, get_indices_constant_regular,\
    get_irregular_indices_grid, get_irregular_neighsmatrix,\
    get_regular_neighsmatrix, get_relative_neighs, get_indices_from_borders,\
    new_get_borders_from_irregular_extremes, new_get_irregular_extremes,\
    get_core_indices, get_extremes_regularneighs_grid,\
    generate_grid_neighs_coord, generate_grid_neighs_coord_i

from pySpatialTools.Retrieve import create_retriever_input_output

## Aux_retriever
from pySpatialTools.Retrieve.aux_retriever import NullRetriever,\
    _check_retriever, create_retriever_input_output,\
    _general_autoexclude, _array_autoexclude, _list_autoexclude
from pySpatialTools.Retrieve import DummyRetriever
## Tools retriever
from pySpatialTools.Retrieve.tools_retriever import create_aggretriever
from pySpatialTools.SpatialRelations import DummyRegDistance
from pySpatialTools.Discretization import GridSpatialDisc

#from scipy.sparse import coo_matrix

from pySpatialTools.utils.artificial_data import \
    random_transformed_space_points, generate_random_relations_cutoffs
from pySpatialTools.Discretization import SetDiscretization
from pySpatialTools.Retrieve.aux_windowretriever import windows_iteration
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    NonePerturbation, JitterLocations, PermutationIndPerturbation,\
    ContiniousIndPerturbation, DiscreteIndPerturbation,\
    MixedFeaturePertubation, PermutationPerturbationLocations


def test():
    ## Parameters
    n = 100
    # Implicit
    locs = np.random.random((n, 2))*100
    locs1 = random_transformed_space_points(n, 2, None)*10
    # Explicit
    disc0 = SetDiscretization(np.random.randint(0, 20, 100))
    input_map = lambda s, x: disc0.discretize(x)
    pars4 = {'order': 4}
    pars5 = {'lim_distance': 2}
    pars8 = {'l': 8, 'center': 0, 'excluded': False}
    mainmapper = generate_random_relations_cutoffs(20, store='sparse')
    mainmapper.set_inout(output='indices')
    inttypes = [int, np.int32, np.int64]

    ###########################################################################
    ######### WindowRetriever module
    shape = (10, 10)
    # creation windowretriever
    map2indices, map2locs, WindowRetriever = create_window_utils((10, 10))
    windret = WindowRetriever(shape, map2indices, map2locs)
    windret.data
    windret.get_indices(np.array([[0, 0]]))
    windret.get_locations([0, 1])
    len(windret)
    try:
        boolean = False
        map2locs(-1)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    map2indices(5)

    map2indices = create_map2indices((10, 10))
    map2indices(5)
    map2indices(np.array([[0, 0]]))

    # windows_iteration
    shape, max_bunch, l, center, excluded = (10, 10), 20, 3, -1, False
    shapes = np.array(list(np.cumprod(shape[1:][::-1])[::-1]) + [1])
    coord = np.random.randint(0, 10, 2).reshape((1, 2))
    for i, nei, rp in windows_iteration(shape, max_bunch, l, center, excluded):
        pass
    for i, nei, rp in windows_iteration(shape, max_bunch, l, center, True):
        pass

    indices, relative_neighs, rel_pos =\
        get_indices_constant_regular(shape, map2indices, l, center, excluded)
    extremes, ranges, sizes =\
        get_irregular_indices_grid(shape, l, center, excluded)
    neighs = get_irregular_neighsmatrix(indices, relative_neighs, shapes)
    neighs = get_regular_neighsmatrix(indices, relative_neighs)

    borders, ranges, sizes_nei =\
        get_irregular_indices_grid(shape, l, center, excluded)
    diff = get_relative_neighs(shape, [l]*2, [center]*2, excluded)
    points, ranges = new_get_irregular_extremes(diff, shape)
    get_extremes_regularneighs_grid(shape, l, center, excluded)
    get_extremes_regularneighs_grid(shape, l, center, True)

#    indices = get_indices_from_borders(borders, map2indices)
#    points_corners, ranges =\
#        new_get_borders_from_irregular_extremes(borders, shape, ranges)
#
#    indices = get_core_indices(borders, map2indices)

    generate_grid_neighs_coord(coord, shape, 2, l, center, excluded)
    generate_grid_neighs_coord_i(coord, shape, 2, l, center, excluded)
    generate_grid_neighs_coord_i(coord, shape, 2, l, center, True)
    try:
        boolean = False
        aux = np.random.randint(0, 10, 4).reshape((1, 4))
        generate_grid_neighs_coord_i(aux, shape, 2, l, center, True)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ###########################################################################
    ######### Aux_Retriever module
    # parameters
    n_iss, n_neighs = 4, 5
    to_exclude_elements = np.random.randint(0, 10, n_iss).reshape((n_iss, 1))
    neighs = np.random.randint(0, 10, n_iss*n_neighs)
    neighs = neighs.reshape((n_iss, n_neighs))
    dists = np.random.random((n_iss, n_neighs, 2))
    regions = np.random.randint(0, 10, 100)
    idxs = to_exclude_elements

    # Testing
    dummyret = NullRetriever(regions)
    try:
        boolean = False
        _check_retriever(dummyret)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    dummyret = NullRetriever(regions)
    dummyret.retriever = None
    dummyret._default_ret_val = None
    # Testing
    dummyret = NullRetriever(regions)
    try:
        boolean = False
        _check_retriever(dummyret)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    dummyret = NullRetriever(regions)
    dummyret._define_retriever = None
    dummyret._format_output_exclude = None
    dummyret._format_output_noexclude = None
    # Testing
    dummyret = NullRetriever(regions)
    try:
        boolean = False
        _check_retriever(dummyret)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    # Auxiliar exlude functions
    _list_autoexclude(to_exclude_elements, neighs, dists)
    _array_autoexclude(to_exclude_elements, neighs, dists)
    _general_autoexclude(to_exclude_elements, neighs, dists)
    _general_autoexclude(to_exclude_elements, list(neighs), list(dists))
    _list_autoexclude(to_exclude_elements, neighs, None)
    _array_autoexclude(to_exclude_elements, neighs, None)
    _general_autoexclude(to_exclude_elements, neighs, None)
    _general_autoexclude(to_exclude_elements, list(neighs), None)

    # map regions to points
    mapin_regpoints, mapout_regpoints = create_retriever_input_output(regions)
    mapout_regpoints(None, idxs, (neighs, dists))

    ###########################################################################
    ######### Tools_Retriever module
    disc1 = GridSpatialDisc((10, 10), xlim=(0, 1), ylim=(0, 1))
    regions = np.random.randint(0, 50, 100)
    ret = create_aggretriever(regions, regmetric=None)
    ret._input_map(0)
    try:
        boolean = False
        ret._input_map(None)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    ret = create_aggretriever(disc1, DummyRegDistance(regions))
    ret._input_map(np.random.random((1, 2)))
    create_aggretriever(disc1, retriever=SameEleNeigh)

    ###########################################################################
    ######### Exhaustive testing over common retrievers tools
    ## Perturbations
    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
    k_perturb4 = k_perturb1+k_perturb2+k_perturb3
    ## Create perturbations
    reind = np.vstack([np.random.permutation(n) for i in range(k_perturb1)])
    perturbation1 = PermutationPerturbation(reind.T)
    perturbation2 = NonePerturbation(k_perturb2)
#    perturbation3 = JitterLocations(0.2, k_perturb3)
    reind = np.vstack([np.random.permutation(n) for i in range(k_perturb3)])
    perturbation3 = PermutationPerturbationLocations(reind.T)
    perturbation4 = [perturbation1, perturbation2, perturbation3]
    _input_map = lambda s, i: i
    _output_map = [lambda s, i, x: x]
    ## Possibilities
    pos_autodata = [True, None, np.arange(n).reshape((n, 1))]
    pos_inmap = [None, _input_map]
    pos_outmap = [None, _output_map, _output_map[0]]
    pos_inforet = [None, 0, lambda x, pars: 0]
    pos_infof = [None, lambda x, pars: 0]
    pos_constantinfo = [True, False, None]
    pos_typeret = ['space', '']
    pos_perturbations = [None, perturbation4]
    pos_ifdistance = [True, False]
    pos_autoexclude = [False]  # True, None for other time
    pos_relativepos = [None]
    pos_boolinidx = [True, False]
    pos_preferable_input = [True, False]
    pos_constantneighs = [True, False, None]
    pos_listind = [True, False, None]

    possibles = [pos_autodata, pos_inmap, pos_outmap, pos_inforet, pos_infof,
                 pos_constantinfo, pos_typeret, pos_perturbations,
                 pos_ifdistance, pos_autoexclude, pos_relativepos,
                 pos_boolinidx, pos_preferable_input, pos_constantneighs,
                 pos_listind]
    counter = -1
    for p in product(*possibles):
        counter += 1
##        print p, counter
        ret = DummyRetriever(n, autodata=p[0], input_map=p[1], output_map=p[2],
                             info_ret=p[3], info_f=p[4], constant_info=p[5],
                             perturbations=p[7], autoexclude=p[9],
                             ifdistance=p[8], relative_pos=p[10],
                             bool_input_idx=p[11], typeret=p[6],
                             preferable_input_idx=p[12], constant_neighs=p[13],
                             bool_listind=p[14])
        ## Selecting point_i
        if p[11] is False:
            i = np.array([0])
            j = [np.array([0]), np.array([1])]
        else:
            i = 0
            j = [0, 1]
        ## Testing functions standards
        ## Get Information
        ################
        # Assert information getting
        info_i, info_i2 = ret._get_info_i(i, 0), ret._get_info_i(j, 0)
        assert(info_i == 0)
        assert(info_i2 == 0)
        ## Get locations
        ################
        if p[11]:
            loc_i = ret.get_loc_i([0])
        else:
            loc_i = ret.get_loc_i([np.array([0])])
#        print loc_i, counter, ret.get_loc_i, p[11]
        assert(len(loc_i) == 1)
        assert(type(loc_i) == type(ret.data_input))
        assert(type(loc_i[0]) == np.ndarray)
        assert(all(loc_i[0] == np.array([0])))
        if p[11]:
            loc_i = ret.get_loc_i([0, 1])
        else:
            loc_i = ret.get_loc_i([np.array([0]), np.array([1])])
#        print loc_i, ret.get_loc_i, p[11]
        assert(len(loc_i) == 2)
        assert(type(loc_i) == type(ret.data_input))
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(all(loc_i[0] == np.array([0])))
        assert(all(loc_i[1] == np.array([1])))
        ## Get indices
        ################
        if p[11]:
            i_loc = ret.get_indice_i([0])
        else:
            i_loc = ret.get_indice_i([np.array([0])])
#        print i_loc, counter, ret.get_indice_i, p[11]
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        if p[11]:
            i_loc = ret.get_indice_i([0, 1])
        else:
            i_loc = ret.get_indice_i([np.array([0]), np.array([1])])
#        print i_loc, ret.get_indice_i, p[11]
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
        assert(i_loc[0] == 0)
        assert(i_loc[1] == 1)
        ## Preparing input
        ##################
        # Assert element getting
        e1, e2 = ret._prepare_input(i, 0), ret._prepare_input(j, 0)
#        print i, j, e1, e2, p[11], p[12], ret._prepare_input, ret.get_indice_i
#        print ret.preferable_input_idx
        if p[12]:
            assert(e1 == [0])
            assert(e2 == [0, 1])
        else:
            assert(e1 == [np.array([0])])
#            print e1, ret._prepare_input, p[11], p[12]
            assert(all([type(e) == np.ndarray for e in e1]))
#            print e2, type(e2[0]), ret._prepare_input, p[11], p[12], counter
            assert(np.all([e2 == [np.array([0]), np.array([1])]]))
            assert(all([type(e) == np.ndarray for e in e2]))
        ## Retrieve and output
        ######################
        # Assert correct retrieving
        ## Retrieve individual
        neighs, dists = ret._retrieve_neighs_general_spec(i, 0, p[8])
#        print dists, type(dists), p[8]
        assert(type(neighs[0][0]) in inttypes)
        assert(dists is None or not p[8] is False)
#        print dists, type(dists), p[8]
        ## Output map
        neighs2, dists2 = ret._output_map[0](ret, i, (neighs, dists))
        assert(type(neighs2[0][0]) in inttypes)
        assert(dists2 is None or not p[8] is False)
        ## Output
        neighs, dists = ret._format_output(i, neighs, dists)
        if p[9]:
            print neighs, dists, ret._exclude_auto, i, counter
            assert(len(neighs) == 1)
            assert(len(neighs[0]) == 0)
    #        assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[8] is False)
        else:
            assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[8] is False)
        ## Retrieve multiple
        neighs, dists = ret._retrieve_neighs_general_spec(j, 0, p[8])
        assert(type(neighs[0][0]) in inttypes)
        assert(dists is None or not p[8] is False)
#        print neighs, p, counter
#        print ret.staticneighs, type(neighs[0][0])
        ## Output map
        neighs2, dists2 = ret._output_map[0](ret, i, (neighs, dists))
        assert(type(neighs2[0][0]) in inttypes)
        assert(dists2 is None or not p[8] is False)
        ## Output
        neighs, dists = ret._format_output(j, neighs, dists)
        if p[9]:
            assert(len(neighs) == 2)
            assert(len(neighs[0]) == 0)
            assert(len(neighs[1]) == 0)
    #        assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[8] is False)
        else:
            assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[8] is False)

        if p[5]:
            neighs_info = ret.retrieve_neighs(i)
            neighs_info.get_information()
            neighs_info = ret[i]
            neighs_info.get_information()
        else:
            neighs_info = ret.retrieve_neighs(i, p[3])
            neighs_info.get_information()

        if np.random.random() < 0.1:
            len(ret)
            ret.export_neighs_info()
            if not ret._heterogenous_input:
                ret._n0
            if not ret._heterogenous_output:
                ret._n1
            ret.shape
            ret.data_input
            ret.data_output

        if np.random.random() < 0.1:
            ## Iterations
            ret.set_iter()
            for iss, nei in ret:
                break

#    ###########################################################################
#    ######### Preparation parameters for general testing
#    ## Perturbations
#    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
#    k_perturb4 = k_perturb1+k_perturb2+k_perturb3
#    ## Create perturbations
#    reind = np.vstack([np.random.permutation(n) for i in range(k_perturb1)])
#    perturbation1 = PermutationPerturbation(reind.T)
#    perturbation2 = NonePerturbation(k_perturb2)
#    perturbation3 = JitterLocations(0.2, k_perturb3)
#    perturbation4 = [perturbation1, perturbation2, perturbation3]
#    pos_perturbations = [None, perturbation1, perturbation2, perturbation3,
#                         perturbation4]
#
#    _input_map = lambda s, i: i
#    _output_map = [lambda s, i, x: x]
#    pos_ifdistance = [True, False]
#    pos_inmap = [None, _input_map]
#    pos_constantinfo = [True, False, None]
#    pos_boolinidx = [True, False, None]
#
#    ###########################################################################
#    #### KRetriever
#    ##################
#    pos_inforet = [2, 5, 10]
#    pos_outmap = [None, _output_map]
#
#    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
#           pos_constantinfo, pos_boolinidx, pos_perturbations]
#    for p in product(*pos):
#        ret = KRetriever(locs, info_ret=p[0], ifdistance=p[1], input_map=p[2],
#                         output_map=p[3], constant_info=p[4],
#                         bool_input_idx=p[5], perturbations=p[6])
#        if p[5] is False:
#            i = locs[0]
#        else:
#            i = 0
##        print i, p, ret.staticneighs, ret.neighs_info.staticneighs
#        if p[4]:
#            neighs_info = ret.retrieve_neighs(i)
#            neighs_info.get_information()
#            #neighs_info = ret[i]
#            #neighs_info.get_information()
#        else:
#            neighs_info = ret.retrieve_neighs(i, p[0])
#            neighs_info.get_information()
#
#        ## Testing other functions and parameters
#        ret.k_perturb
#
##    ## Iterations
##    ret.set_iter()
##    for iss, nei in ret:
##        pass
#
#    ###########################################################################
#    #### CircRetriever
#    ##################
#    pos_inforet = [2., 5., 10.]
#    pos_outmap = [None, _output_map]
#
#    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
#           pos_constantinfo, pos_boolinidx]
#    for p in product(*pos):
#        ret = KRetriever(locs, info_ret=p[0], ifdistance=p[1], input_map=p[2],
#                         output_map=p[3], constant_info=p[4],
#                         bool_input_idx=p[5])
#        if p[5] is False:
#            i = locs[0]
#        else:
#            i = 0
##        print i, p
#        if p[4]:
#            neighs_info = ret.retrieve_neighs(i)
#            neighs_info.get_information()
#            #neighs_info = ret[i]
#            #neighs_info.get_information()
#        else:
#            neighs_info = ret.retrieve_neighs(i, p[0])
#            neighs_info.get_information()
#
##    ## Iterations
##    ret.set_iter()
##    for iss, nei in ret:
##        pass
#
#    ###########################################################################
#    #### WindowsRetriever
#    #####################
#    pos_inforet = [{'l': 1, 'center': 0, 'excluded': False},
#                   {'l': 4, 'center': 0, 'excluded': False},
#                   {'l': 3, 'center': 1, 'excluded': True}]
#    pos_outmap = [None, _output_map]
#    shape = 10, 10
#    gridlocs = np.random.randint(0, np.prod(shape), 2000).reshape((1000, 2))
#
#    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
#           pos_constantinfo, pos_boolinidx, pos_perturbations]
#    for p in product(*pos):
#        ret = WindowsRetriever(shape, info_ret=p[0], ifdistance=p[1],
#                               input_map=p[2], output_map=p[3],
#                               constant_info=p[4], bool_input_idx=p[5],
#                               perturbations=p[6])
#        if p[5] is False:
#            i = gridlocs[0].reshape((1, len(shape)))
#        else:
#            i = 0
##        print i, p, ret.staticneighs, ret.neighs_info.staticneighs
#        if p[4]:
#            neighs_info = ret.retrieve_neighs(i)
#            neighs_info.get_information()
#            #neighs_info = ret[i]
#            #neighs_info.get_information()
#        else:
#            neighs_info = ret.retrieve_neighs(i, p[0])
#            neighs_info.get_information()
#
#        ## Testing other functions and parameters
#        ret.k_perturb

#    ## Iterations
#    ret.set_iter()
#    for iss, nei in ret:
#        pass

#    ###########################################################################
#    #### SameEleRetriever
#    #####################
#    pos_inforet = [None]
#    pos_outmap = [None, _output_map]
#
#    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
#           pos_constantinfo, pos_boolinidx]
#    for p in product(*pos):
#        ret = SameEleNeigh(mainmapper, info_ret=p[0], ifdistance=p[1],
#                           input_map=p[2], output_map=p[3], constant_info=p[4],
#                           bool_input_idx=p[5])
#        if p[5] is False:
#            i = mainmapper.data[0]
#            j = mainmapper.data[[0, 3]]
#        else:
#            i = 0
#            j = [0, 3]
#        print i, p
#        if p[4]:
#            neighs_info = ret.retrieve_neighs(i)
#            neighs_info.get_information()
#            #neighs_info = ret[i]
#            #neighs_info.get_information()
#        else:
#            neighs_info = ret.retrieve_neighs(i, p[0])
#            neighs_info.get_information()
#            neighs_info = ret.retrieve_neighs(j, p[0])
#            neighs_info.get_information()
#
##    ## Iterations
##    ret.set_iter()
##    for iss, nei in ret:
##        pass
#
#    ###########################################################################
#    #### OrderEleRetriever
#    ######################
#    pos_inforet = [pars4]
#    pos_outmap = [None, _output_map]
#
#    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
#           pos_constantinfo, pos_boolinidx]
#    for p in product(*pos):
#        ret = OrderEleNeigh(mainmapper, info_ret=p[0], ifdistance=p[1],
#                            input_map=p[2], output_map=p[3],
#                            constant_info=p[4], bool_input_idx=p[5])
#        if p[5] is False:
#            i = mainmapper.data[0]
#            j = mainmapper.data[[0, 3]]
#        else:
#            i = 0
#            j = [0, 3]
#        print i, p
#        if p[4]:
#            neighs_info = ret.retrieve_neighs(i)
#            neighs_info.get_information()
#            #neighs_info = ret[i]
#            #neighs_info.get_information()
#            neighs_info = ret.retrieve_neighs(j)
#            neighs_info.get_information()
#        else:
#            neighs_info = ret.retrieve_neighs(i, p[0])
#            neighs_info.get_information()
#            neighs_info = ret.retrieve_neighs(j, p[0])
#            neighs_info.get_information()
#
##    ## Iterations
##    ret.set_iter()
##    for iss, nei in ret:
##        pass
#
#    ###########################################################################
#    #### LimDistanceRetriever
#    ##########################
#    pos_inforet = [pars5]
#    pos_outmap = [None, _output_map]
#
#    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
#           pos_constantinfo, pos_boolinidx]
#    for p in product(*pos):
#        ret = LimDistanceEleNeigh(mainmapper, info_ret=p[0], ifdistance=p[1],
#                                  input_map=p[2], output_map=p[3],
#                                  constant_info=p[4], bool_input_idx=p[5])
#        if p[5] is False:
#            i = mainmapper.data[0]
#            j = mainmapper.data[[0, 3]]
#        else:
#            i = 0
#            j = [0, 3]
#        print i, p
#        if p[4]:
#            neighs_info = ret.retrieve_neighs(i)
#            neighs_info.get_information()
#            #neighs_info = ret[i]
#            #neighs_info.get_information()
#            neighs_info = ret.retrieve_neighs(j)
#            neighs_info.get_information()
#        else:
#            neighs_info = ret.retrieve_neighs(i, p[0])
#            neighs_info.get_information()
#            neighs_info = ret.retrieve_neighs(j, p[0])
#            neighs_info.get_information()
#
##    ## Iterations
##    ret.set_iter()
##    for iss, nei in ret:
##        pass

##info_ret=None, autolocs=None, pars_ret=None,
##                 autoexclude=True, ifdistance=False, info_f=None,
##                 perturbations=None, relative_pos=None, input_map=None,
##                 output_map=None, constant_info=False, bool_input_idx=None,
##                 format_level=None, type_neighs=None, type_sp_rel_pos=None
#
#    ## Implicit
#    ret0 = KRetriever(locs, 3, ifdistance=True, input_map=_input_map,
#                      output_map=_output_map)
#    ret1 = CircRetriever(locs, 3, ifdistance=True, bool_input_idx=True)
#    ret2 = KRetriever(locs1, 3, ifdistance=True, bool_input_idx=True)
#
#    ## Explicit
#    ret3 = SameEleNeigh(mainmapper, input_map=input_map,
#                        bool_input_idx=True)
#    ret4 = OrderEleNeigh(mainmapper, pars4, input_map=input_map,
#                         bool_input_idx=True)
#    ret5 = LimDistanceEleNeigh(mainmapper, pars5, input_map=input_map,
#                               bool_input_idx=True)
#
#    info_f = lambda x, y: 2
#    relative_pos = lambda x, y: y
#    ret6 = KRetriever(locs1, 3, ifdistance=True, constant_info=True,
#                      autoexclude=False, info_f=info_f, bool_input_idx=True,
#                      relative_pos=relative_pos)
#    ret7 = KRetriever(locs1, 3, ifdistance=True, constant_info=True,
#                      autoexclude=False, info_f=info_f, bool_input_idx=False,
#                      relative_pos=relative_pos)
#    ret8 = WindowsRetriever((100,), pars_ret=pars8)
#
#    ## Retriever Manager
#    gret = RetrieverManager([ret0, ret1, ret2, ret3, ret4, ret5])
#
#    for i in xrange(n):
#        ## Reduce time of computing
#        if np.random.random() < 0.8:
#            continue
#        print 'xz'*80, i
#        neighs_info = ret0.retrieve_neighs(i)
#        neighs_info = ret1.retrieve_neighs(i)
#        neighs_info = ret2.retrieve_neighs(i)
#        neighs_info = ret3.retrieve_neighs(i)
#        neighs_info = ret4.retrieve_neighs(i)
#        neighs_info = ret5.retrieve_neighs(i)
#        neighs_info = ret6.retrieve_neighs(i)
#        neighs_info = ret7.retrieve_neighs(locs1[i])
#        neighs_info = gret.retrieve_neighs(i)
#        neighs_info = ret8.retrieve_neighs(i)
#
##    ret0._prepare_input = lambda x, kr:
##    neighs_info = ret0._retrieve_neighs_dynamic(0)
#
#    neighs_info = ret1._retrieve_neighs_constant_nodistance(4)
#    neighs_info = ret1._retrieve_neighs_constant_distance(4)
#    neighs_info = ret2._retrieve_neighs_constant_nodistance(4)
#    neighs_info = ret8._retrieve_neighs_constant_nodistance(8, pars8)
#    neighs_info = ret8._retrieve_neighs_constant_distance(8, pars8)
#
#    ## Retrieve-driven testing
#    for idx, neighs in ret1:
#            pass
#    for idx, neighs in ret3:
#            pass
#    for idx, neighs in ret4:
#            pass
#    for idx, neighs in ret5:
#            pass
#    ret6.set_iter(2, 1000)
#    for idx, neighs in ret6:
#        pass
#
#    ret8.set_iter()
##    for idx, neighs in ret8:
##        pass
##        print idx, neighs
#
#    ## Main functions
#    ret1.data_input
#    ret1.data_output
#    ret1.shape
#    ret1[0]
#
#    ret2.data_input
#    ret2.data_output
#    ret2.shape
#    ret2[0]
#
#    reindices = np.vstack([np.random.permutation(n) for i in range(5)])
#    perturbation = PermutationPerturbation(reindices.T)
#    ret0.add_perturbations(perturbation)
#
#    ##
#    ### TODO: __iter__
##    net = ret1.compute_neighnet()
##    net = ret2.compute_neighnet()
#
#    ## Other external functions
#    aux = np.random.randint(0, 100, 1000)
#    m_in, m_out = create_retriever_input_output(aux)
#
#    dummyret = DummyRetriever(None)
#    try:
#        _check_retriever(dummyret)
#        raise Exception
#    except:
#        pass
#    dummyret.retriever = None
#    dummyret._default_ret_val = None
#    try:
#        _check_retriever(dummyret)
#        raise Exception
#    except:
#        pass
#    dummyret = DummyRetriever(None)
#    dummyret._retrieve_neighs_spec = None
#    dummyret._define_retriever = None
#    dummyret._format_output_exclude = None
#    dummyret._format_output_noexclude = None
#    try:
#        _check_retriever(dummyret)
#        raise Exception
#    except:
#        pass
#
#    ## General Retriever
#    class PruebaRetriever(GeneralRetriever):
#        preferable_input_idx = True
#        auto_excluded = True
#        constant_neighs = True
#        bool_listind = True
#
#        def __init__(self, autoexclude=True, ifdistance=False):
#            bool_input_idx = True
#            info_ret, info_f, constant_info = None, None, None
#            self._initialization()
#            self._format_output_information(autoexclude, ifdistance, None)
#            self._format_exclude(bool_input_idx, self.constant_neighs)
#            self._format_retriever_info(info_ret, info_f, constant_info)
#            ## Format retriever function
#            self._format_retriever_function()
#            self._format_getters(bool_input_idx)
#            self._format_preparators(bool_input_idx)
#            self._format_neighs_info(bool_input_idx, 2, 'list', 'list')
#
#        def _define_retriever(self):
#            pass
#
#        def _retrieve_neighs_general_spec(self, point_i, p, ifdistance=False,
#                                          kr=0):
#            return [[0]], None
#    pruebaret = PruebaRetriever(True)
#    pruebaret.retrieve_neighs(0)
#    pruebaret.retrieve_neighs(1)
#    pruebaret = PruebaRetriever(False)
#    pruebaret.retrieve_neighs(0)
#    pruebaret.retrieve_neighs(1)
#
#    ### Auxiliar functions of window retriever
#    def iteration_auxiliar(shape, l, center, excluded):
#        matrix = np.zeros((shape)).astype(int)
#        matrix = matrix.ravel()
#        for inds, neighs, d in windows_iteration(shape, 1000, l, center,
#                                                 excluded):
#            matrix[inds] += len(neighs)
#            assert(np.all(inds >= 0))
#        matrix = matrix.reshape(shape)
#        #import matplotlib.pyplot as plt
#        #plt.imshow(matrix)
#        #plt.show()
#    shape, l, center, excluded = (10, 10), [4, 5], [0, 0], False
#    iteration_auxiliar(shape, l, center, excluded)
#    shape, l, center, excluded = (10, 10), [2, 5], [2, -1], False
#    iteration_auxiliar(shape, l, center, excluded)
#    shape, l, center, excluded = (10, 10), [4, 3], [-2, -1], True
#    iteration_auxiliar(shape, l, center, excluded)
#    shape, l, center, excluded = (10, 10), [1, 5], [2, 2], False
#    iteration_auxiliar(shape, l, center, excluded)
