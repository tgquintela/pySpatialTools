
"""
test retrievers
---------------
test for retrievers precoded and framework of retrievers.

"""

import numpy as np
from itertools import product

# Auxiliars
from pySpatialTools.utils.neighs_info import Neighs_Info
from pySpatialTools.SpatialRelations.relative_positioner import\
    metric_distances, BaseRelativePositioner

## Retrievers
from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    RetrieverManager, SameEleNeigh, OrderEleNeigh, LimDistanceEleNeigh,\
    DummyRetriever, GeneralRetriever, WindowsRetriever
from pySpatialTools.Retrieve.retrievers import BaseRetriever
from pySpatialTools.Retrieve.aux_retriever import _check_retriever

## WindowsRetriever functions
from pySpatialTools.Retrieve.aux_windowretriever import create_window_utils,\
    windows_iteration, create_map2indices, get_indices_constant_regular,\
    get_irregular_indices_grid, get_irregular_neighsmatrix,\
    get_regular_neighsmatrix, get_relative_neighs, get_indices_from_borders,\
    new_get_borders_from_irregular_extremes, new_get_irregular_extremes,\
    get_core_indices, get_extremes_regularneighs_grid,\
    generate_grid_neighs_coord, generate_grid_neighs_coord_i

## Aux_retriever
from pySpatialTools.Retrieve.aux_retriever import NullCoreRetriever,\
    _check_retriever, _general_autoexclude, _array_autoexclude,\
    _list_autoexclude
from pySpatialTools.Retrieve import DummyRetriever, DummyLocObject
from pySpatialTools.Retrieve import _retriever_parsing_creation
## Tools retriever
from pySpatialTools.Retrieve.tools_retriever import create_aggretriever,\
    dummy_implicit_outretriver, dummy_explicit_outretriver,\
    avgregionlocs_outretriever
from pySpatialTools.SpatialRelations import DummyRegDistance
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve.aux_retriever_parsing import\
    create_m_in_inverse_discretization, create_m_in_direct_discretization,\
    create_m_out_inverse_discretization, create_m_out_direct_discretization


#from scipy.sparse import coo_matrix
from pySpatialTools.utils.artificial_data import \
    random_transformed_space_points, generate_random_relations_cutoffs
from pySpatialTools.Discretization import SetDiscretization
from pySpatialTools.Retrieve.aux_windowretriever import windows_iteration
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    NonePerturbation, JitterLocations, PermutationIndPerturbation,\
    ContiniousIndPerturbation, DiscreteIndPerturbation,\
    MixedFeaturePertubation, PermutationPerturbationLocations
from pySpatialTools.Discretization import _discretization_parsing_creation


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
    mainmapper = generate_random_relations_cutoffs(20, p0=1, store='sparse')
    mainmapper.set_inout(output='indices')
    mainmapper1 = generate_random_relations_cutoffs(20, p0=.1, store='sparse')
    mainmapper1.set_inout(output='indices')
    mainmapper1.set_distanceorweighs(False)
    mainmapper2 = generate_random_relations_cutoffs(20, p0=.1, store='sparse')
    mainmapper2.set_inout(output='indices')
    mainmapper2.set_distanceorweighs(True)
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
        raise Exception("It has to halt here.")
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
        raise Exception("It has to halt here.")
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
    dummyret = NullCoreRetriever(regions)
    try:
        boolean = False
        _check_retriever(dummyret)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    dummyret = NullCoreRetriever(regions)
    dummyret.retriever = None
    dummyret._default_ret_val = None
    # Testing
    dummyret = NullCoreRetriever(regions)
    try:
        boolean = False
        _check_retriever(dummyret)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    dummyret = NullCoreRetriever(regions)
    dummyret._define_retriever = None
    dummyret._format_output_exclude = None
    dummyret._format_output_noexclude = None
    # Testing
    dummyret = NullCoreRetriever(regions)
    try:
        boolean = False
        _check_retriever(dummyret)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    # Auxiliar exlude functions
    def ensure_neighs_info_instance(neighs, dists):
        # Neighs info instantiation
        type_neighs = 'list' if type(neighs) == list else 'array'
        type_sp_rel_pos = 'list' if type(dists) == np.ndarray else 'array'
        type_sp_rel_pos = None if dists is None else type_sp_rel_pos
        nei = Neighs_Info(format_level=2, type_neighs=type_neighs,
                          type_sp_rel_pos=type_sp_rel_pos, staticneighs=True)
        iss = range(len(neighs))
        nei.set((neighs, dists), iss)

    def ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, dists,
                              flag=False):
        assert(len(to_exclude_elements) == len(neighs))
        assert(len(to_exclude_elements) == len(out))
        if dists is not None:
            assert(len(to_exclude_elements) == len(dists))
            assert(len(out) == len(o_d))
            assert(all([len(out[i]) == len(o_d[i]) for i in range(len(out))]))
#        print to_exclude_elements, out, len(to_exclude_elements), len(out)
        ## Ensure instantiation
        ensure_neighs_info_instance(neighs, dists)
#        print out, o_d
        if flag:
            out, o_d = list(out), list(o_d)
            ensure_neighs_info_instance(out, o_d)

    # Normal random neighs, dists
    out, o_d = _list_autoexclude(to_exclude_elements, neighs, dists)
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, dists)
    out, o_d = _array_autoexclude(to_exclude_elements, neighs, dists)
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, dists, True)
    out, o_d = _general_autoexclude(to_exclude_elements, neighs, dists)
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, dists)
    out, o_d = _general_autoexclude(to_exclude_elements, list(neighs),
                                    list(dists))
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, dists)
    # dists with None
    out, o_d = _list_autoexclude(to_exclude_elements, neighs, None)
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, None)
    out, o_d = _array_autoexclude(to_exclude_elements, neighs, None)
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, None)
    out, o_d = _general_autoexclude(to_exclude_elements, neighs, None)
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, None)
    out, o_d = _general_autoexclude(to_exclude_elements, list(neighs), None)
    ensure_proper_exclude(out, o_d, to_exclude_elements, neighs, None)
    # Empty neighs and dists
    out, o_d = _list_autoexclude(neighs, neighs, dists)
    ensure_proper_exclude(out, o_d, neighs, neighs, dists)
    out, o_d = _array_autoexclude(neighs, neighs, dists)
    ensure_proper_exclude(out, o_d, neighs, neighs, dists)
    out, o_d = _general_autoexclude(neighs, neighs, dists)
    ensure_proper_exclude(out, o_d, neighs, neighs, dists)
    out, o_d = _general_autoexclude(neighs, list(neighs), list(dists))
    ensure_proper_exclude(out, o_d, neighs, neighs, dists)

    # map regions to points
#    mapin_regpoints, mapout_regpoints = create_retriever_input_output(regions)
#    mapout_regpoints(None, idxs, (neighs, dists))
    #### TODO: Other creations testing
    disc1 = GridSpatialDisc((10, 10), xlim=(0, 1), ylim=(0, 1))
    locs = np.random.random((100, 2))
    regs = disc1.discretize(locs)

    m_in_inverse = create_m_in_inverse_discretization((disc1, locs))
    m_in_inverse(None, idxs)
    m_in_inverse = create_m_in_inverse_discretization((locs, regs))
    m_in_inverse(None, idxs)
    m_in_direct = create_m_in_direct_discretization((disc1, locs))
    m_in_direct(None, idxs)
    m_in_direct = create_m_in_direct_discretization((locs, regs))
    m_in_direct(None, idxs)
    m_out_inverse = create_m_out_inverse_discretization((disc1, locs))
    m_out_inverse(None, idxs, (neighs, dists))
    m_out_inverse = create_m_out_inverse_discretization((locs, regs))
    m_out_inverse(None, idxs, (neighs, dists))
    m_out_direct = create_m_out_direct_discretization((disc1, locs))
    m_out_direct(None, idxs, (neighs, dists))
    m_out_direct = create_m_out_direct_discretization((locs, regs))
    m_out_direct(None, idxs, (neighs, dists))
    m_out_inverse = create_m_out_inverse_discretization((disc1, locs))
    m_out_inverse(None, idxs, (neighs, [None]*len(neighs)))
    m_out_inverse = create_m_out_inverse_discretization((locs, regs))
    m_out_inverse(None, idxs, (neighs, [None]*len(neighs)))
    m_out_direct = create_m_out_direct_discretization((disc1, locs))
    m_out_direct(None, idxs, (neighs, [None]*len(neighs)))
    m_out_direct = create_m_out_direct_discretization((locs, regs))
    m_out_direct(None, idxs, (neighs, [None]*len(neighs)))

    ###########################################################################
    ######### Instantiation  (tools_retriever)
    retriever_out = (KRetriever, {'info_ret': 3})
    ret_out = dummy_implicit_outretriver(retriever_out, locs, regs, disc1)
    assert(isinstance(ret_out, BaseRetriever))
    ret_out.retrieve_neighs(0)
    ret_out = avgregionlocs_outretriever(retriever_out, locs, regs, disc1)
    assert(isinstance(ret_out, BaseRetriever))
    ret_out.retrieve_neighs(0)

    f_rand = lambda regs: DummyRegDistance(regs)
    retriever_out = (SameEleNeigh, {}, f_rand, np.arange(10))
    ret_out = dummy_explicit_outretriver(retriever_out, locs, regs, disc1)
    assert(isinstance(ret_out, BaseRetriever))
    ret_out.retrieve_neighs(0)

    ###########################################################################
    ######### Tools_Retriever module
    ### TODO: New tests
#####     disc1 = GridSpatialDisc((10, 10), xlim=(0, 1), ylim=(0, 1))
#####     regions = np.random.randint(0, 50, 100)
#####     ret = create_aggretriever(regions, regmetric=None)
#####     ret._input_map(0)
#####     try:
#####         boolean = False
#####         ret._input_map(None)
#####         boolean = True
#####         raise Exception("It has to halt here.")
#####     except:
#####         if boolean:
#####             raise Exception("It has to halt here.")
#####     ret = create_aggretriever(disc1, DummyRegDistance(regions))
#####     ret._input_map(np.random.random((1, 2)))
#####     create_aggretriever(disc1, retriever=SameEleNeigh)

    ###########################################################################
    ######### Exhaustive testing over common retrievers tools
    ## TODO: Support for autoexclude (problems with space perturbation)
    ## Perturbations
    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
    k_perturb4 = k_perturb1+k_perturb2+k_perturb3
    comp_rel_pos = lambda x, y: np.zeros((len(y), len(y[0]), 1))

    class Comp_Relpos:
        def __init__(self):
            pass

        def compute(self, x, y):
            return comp_rel_pos(x, y)
    comp_class_rel_pos = Comp_Relpos()

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
    pos_constantinfo = [True, False, None]
    pos_perturbations = [None, perturbation4]
    pos_ifdistance = [True, False]
    pos_boolinidx = [True, False]
    pos_preferable_input = [True, False]
    pos_listind = [True, False, None]

    pos_inforet = [None, 0, lambda x, pars: 0]
    pos_infof = [None, lambda x, pars: 0]
    pos_typeret = ['space']  # ''
    pos_autoexclude = [False]  # True, None for other time
    pos_relativepos = [None, BaseRelativePositioner(metric_distances)]
    pos_constantneighs = [True, False, None]

    ## Combinations
    possibles = [pos_autodata, pos_inmap, pos_outmap, pos_constantinfo,
                 pos_perturbations, pos_ifdistance, pos_boolinidx,
                 pos_preferable_input, pos_listind]
    ## Sequencials
    pos_auto_excluded = [True, False, None]
    pos_types = ['array', 'list', 'object', 'listobject']
    pos_typeret = ['space', '']
    pos_constantneighs = [True, False, None]
    pos_autoexclude = [False]  # True, None for other time
    pos_relativepos = [None, comp_rel_pos, comp_class_rel_pos]
    pos_inforet = [None, 0, lambda x, pars: 0, np.zeros(n)]
    pos_infof = [None, lambda x, pars: 0]

    counter = -1
    for p in product(*possibles):
        ## Comtinations
        counter += 1
##        print p, counter
        ## Sequential parameters
        types = pos_types[np.random.randint(0, len(pos_types))]
#        types = 'object'
        auto_excluded = pos_auto_excluded[np.random.randint(0, 3)]
        if types in ['object', 'listobject']:
            typeret = ''
        else:
            typeret = pos_typeret[np.random.randint(0, len(pos_typeret))]
        const = pos_constantneighs[np.random.randint(0, 3)]
        inforet = pos_inforet[np.random.randint(0, len(pos_inforet))]
        infof = pos_infof[np.random.randint(0, len(pos_infof))]
        rel_pos = pos_relativepos[np.random.randint(0, len(pos_relativepos))]
        auto_excl = pos_autoexclude[np.random.randint(0, len(pos_autoexclude))]

        ## Non exhaustive
#        if np.random.random() < 0.25:
#            continue

        ## Forbidden combinations
        if types in ['list', 'object', 'listobject'] and p[8]:
            continue
        if p[6] is False and type(inforet) == np.ndarray:
            continue

        ## Instantiation
        ret = DummyRetriever(n, autodata=p[0], input_map=p[1], output_map=p[2],
                             info_ret=inforet, info_f=infof,
                             constant_info=p[3], perturbations=p[4],
                             autoexclude=auto_excl, ifdistance=p[5],
                             relative_pos=rel_pos, bool_input_idx=p[6],
                             typeret=typeret, preferable_input_idx=p[7],
                             constant_neighs=const, bool_listind=p[8],
                             auto_excluded=auto_excluded, types=types)
        ## Selecting point_i
        if p[6] is False:
            if types == 'listobject':
                i = DummyLocObject(np.array([0]))
                j = [i, DummyLocObject(np.array([1]))]
            else:
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
#        print info_i, info_i2
        assert(info_i == 0)
        assert(np.all(info_i2 == 0))
        ## Get locations
        ################
#        print p[6], types
        if p[6]:
            loc_i = ret.get_loc_i([0])
            if types == 'listobject':
                loc_i = [e.location for e in loc_i]
        else:
            if types == 'listobject':
                retloc = DummyLocObject(np.array([0]))
                loc_i = ret.get_loc_i([retloc])
                loc_i = [e.location for e in loc_i]
            else:
                loc_i = ret.get_loc_i([np.array([0])])
#        print loc_i, counter, ret._get_loc_from_idxs, p[6], p[7]
        assert(len(loc_i) == 1)
#        assert(type(loc_i) == type(ret.data_input))
#        print loc_i, types
        assert(type(loc_i[0]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(all(loc_i[0] == np.array([0])))

        if p[6]:
            loc_i = ret.get_loc_i([0, 1])
            if types == 'listobject':
                loc_i = [e.location for e in loc_i]
        else:
            if types == 'listobject':
                aux = DummyLocObject(np.array([0]))
                loc_i = ret.get_loc_i([aux, DummyLocObject(np.array([1]))])
                loc_i = [e.location for e in loc_i]
            else:
                loc_i = ret.get_loc_i([np.array([0]), np.array([1])])

#        print loc_i, ret.get_loc_i, p[6]
        assert(len(loc_i) == 2)
#        assert(type(loc_i) == type(ret.data_input))
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(len(loc_i[1].shape) == 1)
        assert(all(loc_i[0] == np.array([0])))
        assert(all(loc_i[1] == np.array([1])))
        ## Get indices
        ################
        if p[6]:
            loc_i = [0]
        else:
            if types == 'listobject':
                loc_i = [DummyLocObject(np.array([0]))]
            else:
                loc_i = [np.array([0])]
        i_loc = ret.get_indice_i(loc_i, 0)
#        print i_loc, loc_i, counter, ret.get_indice_i, ret._get_idxs_from_locs
#        print list(ret.data_input)
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        if p[6]:
            i_loc = ret.get_indice_i([0, 1])
        else:
            if types == 'listobject':
                loc_i = [DummyLocObject(np.array([0]))]
                loc_i += [DummyLocObject(np.array([1]))]
            else:
                loc_i = [np.array([0]), np.array([1])]
            i_loc = ret.get_indice_i(loc_i, 0)
#        print i_loc, ret.get_indice_i, p[6]
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
        assert(i_loc[0] == 0)
        assert(i_loc[1] == 1)
        ## Preparing input
        ##################
        ret._general_prepare_input([i], kr=0)
        ret._general_prepare_input(j, kr=0)
        # Assert element getting
#        print i, j, p[6], p[7], ret._prepare_input
        e1, e2 = ret._prepare_input(i, 0), ret._prepare_input(j, 0)
#        print i, j, e1, e2, p[6], p[7], ret._prepare_input, ret.get_indice_i
#        print ret.preferable_input_idx
        if types == 'listobject' and p[7] is not True:
            e1, e2 = [e.location for e in e1], [e.location for e in e2]
        if p[7]:
#            print e1, e2, type(e1), type(e2)
            assert(e1 == [0])
            assert(e2 == [0, 1])
        else:
#            print e1, e2, ret._prepare_input, p[6]
            assert(e1 == [np.array([0])])
#            print e1, ret._prepare_input, p[6], p[7]
            assert(all([type(e) == np.ndarray for e in e1]))
#            print e2, type(e2[0]), ret._prepare_input, p[6], p[7], counter
            assert(np.all([e2 == [np.array([0]), np.array([1])]]))
            assert(all([type(e) == np.ndarray for e in e2]))
        ## Retrieve and output
        ######################
        # Assert correct retrieving
        ## Retrieve individual
        neighs, dists = ret._retrieve_neighs_general_spec(i, 0, p[5])
#        print dists, type(dists), p[5]
        assert(type(neighs[0][0]) in inttypes)
        assert(dists is None or not p[5] is False)
#        print dists, type(dists), p[5]
        ## Output map
        neighs2, dists2 = ret._output_map[0](ret, i, (neighs, dists))
        assert(type(neighs2[0][0]) in inttypes)
        assert(dists2 is None or not p[5] is False)
        ## Output
#        print neighs, dists
        neighs, dists = ret._format_output(i, neighs, dists)
        if auto_excl and not auto_excluded:
#            print neighs, dists, ret._exclude_auto, i, counter
            assert(len(neighs) == 1)
            assert(len(neighs[0]) == 0)
#            assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[5] is False)
        else:
            assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[5] is False)
        neighs, dists = ret._retrieve_neighs_general_spec(i, 0, p[5])
#        print dists, type(dists), p[5]
        assert(type(neighs[0][0]) in inttypes)
        assert(dists is None or not p[5] is False)
#        print dists, type(dists), p[5]
        ## Output map
        neighs2, dists2 = ret._output_map[0](ret, i, (neighs, dists))
        assert(type(neighs2[0][0]) in inttypes)
        assert(dists2 is None or not p[5] is False)
        ## Output
#        print neighs, dists
        neighs, dists = ret._format_output(i, neighs, dists)
        if auto_excl and not auto_excluded:
#            print neighs, dists, ret._exclude_auto, i, counter
            assert(len(neighs) == 1)
            assert(len(neighs[0]) == 0)
#            assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[5] is False)
        else:
            assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[5] is False)
        ## Retrieve multiple
        neighs, dists = ret._retrieve_neighs_general_spec(j, 0, p[5])
        assert(type(neighs[0][0]) in inttypes)
        assert(dists is None or not p[5] is False)
#        print neighs, p, counter
#        print ret.staticneighs, type(neighs[0][0])
        ## Output map
        neighs2, dists2 = ret._output_map[0](ret, i, (neighs, dists))
        assert(type(neighs2[0][0]) in inttypes)
        assert(dists2 is None or not p[5] is False)
        ## Output
        neighs, dists = ret._format_output(j, neighs, dists)
        if auto_excl and not auto_excluded:
            assert(len(neighs) == 2)
            assert(len(neighs[0]) == 0)
            assert(len(neighs[1]) == 0)
    #        assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[5] is False)
        else:
            assert(type(neighs[0][0]) in inttypes)
            assert(dists is None or not p[5] is False)

        if p[3]:
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

        if ret.k_perturb == 0:
            k_option = 1
        else:
            k_options = [-1, 100]
            k_option = k_options[np.random.randint(0, 2)]
        if np.random.random() < 0.1:
            ## Iterations
            ret.set_iter()
            for iss, nei in ret:
                break
            if not ret._constant_ret:
                ## The k has to be between 0 and k_perturb+1
                try:
                    boolean = False
                    ret.retrieve_neighs(0, k=k_option)
                    boolean = True
                    raise Exception("It has to halt here.")
                except:
                    if boolean:
                        raise Exception("It has to halt here.")
            try:
                boolean = False
                ret._map_perturb(k_option)
                boolean = True
                raise Exception("It has to halt here.")
            except:
                if boolean:
                    raise Exception("It has to halt here.")

#### Special cases
    ret = DummyRetriever(n, constant_info=True, bool_input_idx=True,
                         preferable_input_idx=True)
    net = ret.compute_neighnet(mapper=0)
    net = ret.compute_neighnet(mapper=0, datavalue=1.)
    ret = DummyRetriever(n, constant_info=True, bool_input_idx=True,
                         preferable_input_idx=True, ifdistance=False)
    net = ret.compute_neighnet(mapper=0)

    ###########################################################################
    ######### Preparation parameters for general testing
    ## Perturbations
    k_perturb1, k_perturb2, k_perturb3 = 5, 10, 3
    k_perturb4 = k_perturb1+k_perturb2+k_perturb3
    ## Create perturbations
    reind = np.vstack([np.random.permutation(n) for i in range(k_perturb1)])
    perturbation1 = PermutationPerturbation(reind.T)
    perturbation2 = NonePerturbation(k_perturb2)
    perturbation3 = JitterLocations(0.2, k_perturb3)
    perturbation4 = [perturbation1, perturbation2, perturbation3]
    pos_perturbations = [None, perturbation1, perturbation2, perturbation3,
                         perturbation4]

    _input_map = lambda s, i: i
    _output_map = [lambda s, i, x: x]
    pos_ifdistance = [True, False]
    pos_inmap = [None, _input_map]
    pos_constantinfo = [True, False, None]
    pos_boolinidx = [True, False]

    def assert_correctneighs(neighs_info, ifdistance, constant, staticneighs,
                             ks, iss):
        iss = [iss] if type(iss) == int else iss
        assert(type(neighs_info.iss) == list)
        assert(neighs_info.staticneighs == staticneighs)
        if not staticneighs:
#            print neighs_info.ks, ks
            assert(type(neighs_info.ks) == list)
            assert(neighs_info.ks == ks)
        if ifdistance:
            assert(neighs_info.sp_relative_pos is not None)
        else:
            assert(neighs_info.sp_relative_pos is None)
#        print neighs_info.iss, iss, neighs_info.staticneighs
        assert(neighs_info.iss == iss)

    ###########################################################################
    #### KRetriever
    ##################
    pos_inforet = [1, 2, 5, 10]
    pos_outmap = [None, _output_map]
    pos_autoexclude = [False, True]
    pos_pars_ret = [None, 1000]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx, pos_perturbations, pos_autoexclude]
    for p in product(*pos):
        ## Random
        pret = pos_pars_ret[np.random.randint(0, len(pos_pars_ret))]
        ## Instantiation
        ret = KRetriever(locs, info_ret=p[0], ifdistance=p[1], input_map=p[2],
                         output_map=p[3], constant_info=p[4], autoexclude=p[7],
                         bool_input_idx=p[5], perturbations=p[6],
                         pars_ret=pret)
#        print p
        ## Selecting point_i
        if p[5] is False:
            i = locs[0]
            j = [locs[0], locs[1]]
        else:
            i = 0
            j = [0, 1]

        ## Get Information
        ################
        # Assert information getting
        info_i, info_i2 = ret._get_info_i(i, {}), ret._get_info_i(j, {})
#        print info_i, info_i2, ret._default_ret_val, p[0], p, ret._get_info_i
        if p[0] is None:
            assert(info_i == ret._default_ret_val)
            assert(info_i2 == ret._default_ret_val)
        else:
            assert(info_i == p[0])
            assert(info_i2 == p[0])

        ## Get locations
        ################
        loc_i = ret.get_loc_i([i])
        assert(len(loc_i) == 1)
        assert(type(loc_i[0]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(all(loc_i[0] == locs[0]))
        loc_i = ret.get_loc_i(j)
        assert(len(loc_i) == 2)
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(len(loc_i[1].shape) == 1)
        assert(all(loc_i[0] == locs[0]))
        assert(all(loc_i[1] == locs[1]))

        ## Get indices
        ################
        i_loc = ret.get_indice_i([i], 0)
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(i_loc[0] == 0)
        i_loc = ret.get_indice_i(j, 0)
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
#        print i_loc, j
        assert(i_loc[0] == 0)
        assert(i_loc[1] == 1)

#        print i, p, ret.staticneighs, ret.neighs_info.staticneighs
        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            neighs_info.get_information()
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), 0)
            neighs_info = ret[i]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), 0)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), [0, 1])
            neighs_info.get_information()
            neighs_info = ret[j]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), [0, 1])
            neighs_info.get_information()
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), 0)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), [0, 1])
            neighs_info.get_information()

        ## Testing other functions and parameters
        ret.k_perturb

        ## Iterations
        ret.set_iter()
        for iss, nei in ret:
            assert(list(nei.iss) == list(iss))
            break

    ###########################################################################
    #### CircRetriever
    ##################
    pos_inforet = [0.01, 2., 5., 10.]
    pos_outmap = [None, _output_map]
    pos_autoexclude = [False, True]
    pos_pars_ret = [None, 1000]
    pos_ifdistance = [True, False]
    pos_constantinfo = [True, False, None]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx, pos_autoexclude]
    for p in product(*pos):
        ## Random
        pret = pos_pars_ret[np.random.randint(0, len(pos_pars_ret))]

        ## Instantiation
        ret = CircRetriever(locs, info_ret=p[0], ifdistance=p[1],
                            input_map=p[2], output_map=p[3],
                            constant_info=p[4], bool_input_idx=p[5],
                            autoexclude=p[6], pars_ret=pret)
#        print p
        ## Selecting point_i
        if p[5] is False:
            i = locs[0]
            j = [locs[0], locs[1]]
        else:
            i = 0
            j = [0, 1]

        ## Get Information
        ################
        # Assert information getting
        info_i, info_i2 = ret._get_info_i(i, {}), ret._get_info_i(j, {})
#        print info_i, info_i2, ret._default_ret_val, p[0], p, ret._get_info_i
        if p[0] is None:
            assert(info_i == ret._default_ret_val)
            assert(info_i2 == ret._default_ret_val)
        else:
            assert(info_i == p[0])
            assert(info_i2 == p[0])
        ## Get locations
        ################
        loc_i = ret.get_loc_i([i])
        assert(len(loc_i) == 1)
        assert(type(loc_i[0]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(all(loc_i[0] == locs[0]))
        loc_i = ret.get_loc_i(j)
        assert(len(loc_i) == 2)
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(len(loc_i[1].shape) == 1)
        assert(all(loc_i[0] == locs[0]))
        assert(all(loc_i[1] == locs[1]))

        ## Get indices
        ################
        i_loc = ret.get_indice_i([i], 0)
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(i_loc[0] == 0)
        i_loc = ret.get_indice_i(j, 0)
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
        assert(i_loc[0] == 0)
        assert(i_loc[1] == 1)

        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), 0)
            neighs_info.get_information()
            neighs_info = ret[i]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), 0)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), [0, 1])
            neighs_info.get_information()
            neighs_info = ret[j]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), [0, 1])
            neighs_info.get_information()
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j, p[0])
            neighs_info.get_information()

        ## Iterations
        ret.set_iter()
        for iss, nei in ret:
            assert(list(nei.iss) == list(iss))
            break

    ###########################################################################
    #### WindowsRetriever
    #####################
    ## TODO: Relative pos and autoexclude=True
    pos_inforet = [{'l': 1, 'center': 0, 'excluded': False},
                   {'l': 4, 'center': 0, 'excluded': False},
                   {'l': 3, 'center': 1, 'excluded': True}]
    pos_outmap = [None, _output_map]
    shape = 10, 10
    gridlocs = np.random.randint(0, np.prod(shape), 2000).reshape((1000, 2))

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx, pos_perturbations]
    ## Random exploration of inputs space
    pos_relativepos = [None, comp_rel_pos, comp_class_rel_pos]

    for p in product(*pos):
        ## Random exploration of inputs space
        rel_pos = pos_relativepos[np.random.randint(0, len(pos_relativepos))]

        ret = WindowsRetriever(shape, info_ret=p[0], ifdistance=p[1],
                               input_map=p[2], output_map=p[3],
                               constant_info=p[4], bool_input_idx=p[5],
                               perturbations=p[6], relative_pos=rel_pos)
##        print p
        ## Selecting point_i
        ind_i, ind_j = 0, [0, 5]
        if p[5] is False:
            i = ret.retriever[0].data[ind_i]
            j = ret.retriever[0].data[ind_j]
        else:
            i = ind_i
            j = ind_j

        ## Get Information
        ################
        # Assert information getting
        info_i, info_i2 = ret._get_info_i(i, {}), ret._get_info_i(j, {})
        if p[0] is None:
            assert(info_i == ret._default_ret_val)
            assert(info_i2 == ret._default_ret_val)
        else:
            assert(info_i == p[0])
            assert(info_i2 == p[0])

        ## Get locations
        ################
        loc_i = ret.get_loc_i([i])
        assert(len(loc_i) == 1)
        assert(type(loc_i[0]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
#        assert(all(loc_i[0] == locs[0]))
        loc_i = ret.get_loc_i(j)
        assert(len(loc_i) == 2)
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(len(loc_i[1].shape) == 1)
#        assert(all(loc_i[0] == locs[0]))
#        assert(all(loc_i[1] == locs[1]))

        ## Get indices
        ################
        i_loc = ret.get_indice_i([i], 0)
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(i_loc[0] == ind_i)
        i_loc = ret.get_indice_i(j, 0)
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
        assert(i_loc[0] == ind_j[0])
        assert(i_loc[1] == ind_j[1])

        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret[i]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
            neighs_info = ret[j]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()

        ## Iterations
        ret.set_iter()
        for iss, nei in ret:
            assert(list(nei.iss) == list(iss))
            break

    try:
        boolean = False
        ret = WindowsRetriever((10, 10, 10.))
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")

    ###########################################################################
    #### SameEleRetriever
    #####################
    pos_inforet = [None]
    pos_outmap = [None, _output_map]
    pos_autoexclude = [False, True]

    pos_mainmapper = [mainmapper, mainmapper1]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx, pos_autoexclude]

    for p in product(*pos):
        # Random parameter exploration
        i_mapper = np.random.randint(0, len(pos_mainmapper))
        choose_mapper = pos_mainmapper[i_mapper]
        # Instantiation
        ret = SameEleNeigh(choose_mapper, info_ret=p[0], ifdistance=p[1],
                           input_map=p[2], output_map=p[3], constant_info=p[4],
                           bool_input_idx=p[5], autoexclude=p[6])
#        print p
        ## Selecting point_i
        ind_i, ind_j = 0, [0, 5]
        if p[5] is False:
            i = choose_mapper.data_input[ind_i]
            j = choose_mapper.data_input[ind_j]
        else:
            i = ind_i
            j = ind_j
#        print '+'*50, j, choose_mapper.data[ind_j]
#        print choose_mapper.data_input[ind_j]
#        print i_mapper, ret.get_indice_i
#        print ret.preferable_input_idx, p[5], ret._get_idxs_from_locs
#        print ret._get_idx_from_loc

        ## Get Information
        ################
        # Assert information getting
        info_i, info_i2 = ret._get_info_i(i, {}), ret._get_info_i(j, {})
        assert(info_i == {})
        assert(info_i2 == {})

        ## Get locations
        ################
        loc_i = ret.get_loc_i([i])
        assert(len(loc_i) == 1)
        assert(type(loc_i[0]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(all(loc_i[0] == choose_mapper.data_input[ind_i]))
        loc_i = ret.get_loc_i(j)
        assert(len(loc_i) == 2)
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(len(loc_i[1].shape) == 1)
        assert(all(loc_i[0] == choose_mapper.data_input[ind_j[0]]))
        assert(all(loc_i[1] == choose_mapper.data_input[ind_j[1]]))

        ## Get indices
        ################
        i_loc = ret.get_indice_i([i], 0)
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(i_loc[0] == ind_i)
        i_loc = ret.get_indice_i(j, 0)
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
#        print ret.data_input, i_loc, j, p[5]
        assert(i_loc[0] == ind_j[0])
        assert(i_loc[1] == ind_j[1])

        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret[i]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
            neighs_info = ret[j]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()

        ## Iterations
        ret.set_iter()
        for iss, nei in ret:
            assert(list(nei.iss) == list(iss))
            break

    ###########################################################################
    #### LimDistanceRetriever
    ##########################
    pars_lim0, pars_lim1 = pars5, {'lim_distance': None}
    pars_lim2 = {'maxif': False, 'lim_distance': 1}
    pos_inforet = [None, pars_lim0, pars_lim1, pars_lim2]
    pos_outmap = [None, _output_map]
    pos_autoexclude = [False, True]
    pos_mainmapper = [mainmapper, mainmapper1]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx, pos_autoexclude]

    for p in product(*pos):
        # Random parameter exploration
        i_mapper = np.random.randint(0, len(pos_mainmapper))
        choose_mapper = pos_mainmapper[i_mapper]
        # Instantiation
        ret = LimDistanceEleNeigh(choose_mapper, info_ret=p[0],
                                  ifdistance=p[1], input_map=p[2],
                                  output_map=p[3], constant_info=p[4],
                                  bool_input_idx=p[5], autoexclude=p[6])
#        print p
        ## Selecting point_i
        ind_i, ind_j = 0, [0, 5]
        if p[5] is False:
            i = choose_mapper.data_input[ind_i]
            j = choose_mapper.data_input[ind_j]
        else:
            i = ind_i
            j = ind_j

        ## Get Information
        ################
        # Assert information getting
        info_i, info_i2 = ret._get_info_i(i, {}), ret._get_info_i(j, {})
#        print info_i, info_i2, ret._default_ret_val, p[0], p, ret._get_info_i
        if p[0] is None:
            assert(info_i == ret._default_ret_val)
            assert(info_i2 == ret._default_ret_val)
        else:
            assert(info_i == p[0])
            assert(info_i2 == p[0])

        ## Get locations
        ################
        loc_i = ret.get_loc_i([i])
        assert(len(loc_i) == 1)
        assert(type(loc_i[0]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(all(loc_i[0] == choose_mapper.data_input[ind_i]))
        loc_i = ret.get_loc_i(j)
        assert(len(loc_i) == 2)
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(len(loc_i[1].shape) == 1)
        assert(all(loc_i[0] == choose_mapper.data_input[ind_j[0]]))
        assert(all(loc_i[1] == choose_mapper.data_input[ind_j[1]]))

        ## Get indices
        ################
        i_loc = ret.get_indice_i([i], 0)
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(i_loc[0] == ind_i)
        i_loc = ret.get_indice_i(j, 0)
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
        assert(i_loc[0] == ind_j[0])
        assert(i_loc[1] == ind_j[1])

#        print i, p
        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret[i]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
            neighs_info = ret[j]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()

        ## Iterations
        ret.set_iter()
        for iss, nei in ret:
            assert(list(nei.iss) == list(iss))
            break

    ###########################################################################
    #### OrderEleRetriever
    ######################
    pars_order0, pars_order1 = pars4, {'exactorlimit': True, 'order': 2}
    pars_order2 = {'exactorlimit': True, 'order': 4}
    pos_inforet = [None, pars_order0, pars_order1, pars_order2]
    pos_outmap = [None, _output_map]
    pos_autoexclude = [False, True]
    pos_mainmapper = [mainmapper, mainmapper1, mainmapper2]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx, pos_autoexclude]

    for p in product(*pos):
        # Random parameter exploration
        i_mapper = np.random.randint(0, len(pos_mainmapper))
        choose_mapper = pos_mainmapper[i_mapper]
        # Instantiation
        ret = OrderEleNeigh(choose_mapper, info_ret=p[0], ifdistance=p[1],
                            input_map=p[2], output_map=p[3],
                            constant_info=p[4], bool_input_idx=p[5],
                            autoexclude=p[6])
#        print p
        ## Selecting point_i
        ind_i, ind_j = 0, [0, 3]
        if p[5] is False:
            i = choose_mapper.data_input[ind_i]
            j = choose_mapper.data_input[ind_j]
        else:
            i = ind_i
            j = ind_j

        ## Get Information
        ################
        # Assert information getting
        info_i, info_i2 = ret._get_info_i(i, {}), ret._get_info_i(j, {})
#        print info_i, info_i2, ret._default_ret_val, p[0], p, ret._get_info_i
        if p[0] is None:
            assert(info_i == ret._default_ret_val)
            assert(info_i2 == ret._default_ret_val)
        else:
            assert(info_i == p[0])
            assert(info_i2 == p[0])

        ## Get locations
        ################
        loc_i = ret.get_loc_i([i])
        assert(len(loc_i) == 1)
        assert(type(loc_i[0]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(all(loc_i[0] == choose_mapper.data_input[ind_i]))
        loc_i = ret.get_loc_i(j)
        assert(len(loc_i) == 2)
        assert(type(loc_i[0]) == np.ndarray)
        assert(type(loc_i[1]) == np.ndarray)
        assert(len(loc_i[0].shape) == 1)
        assert(len(loc_i[1].shape) == 1)
        assert(all(loc_i[0] == choose_mapper.data_input[ind_j[0]]))
        assert(all(loc_i[1] == choose_mapper.data_input[ind_j[1]]))

        ## Get indices
        ################
        i_loc = ret.get_indice_i([i], 0)
        assert(len(i_loc) == 1)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        i_loc = ret.get_indice_i(j, 0)
        assert(len(i_loc) == 2)
        assert(type(i_loc) == list)
        assert(type(i_loc[0]) in inttypes)
        assert(type(i_loc[1]) in inttypes)
        assert(i_loc[0] == j[0])
        assert(i_loc[1] == j[1])

        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret[i]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j)
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
            neighs_info = ret[j]
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_i)
            neighs_info.get_information()
            neighs_info = ret.retrieve_neighs(j, p[0])
            assert_correctneighs(neighs_info, p[1], p[4], ret.staticneighs,
                                 range(ret.k_perturb+1), ind_j)
            neighs_info.get_information()

        ## Iterations
        ret.set_iter()
        for iss, nei in ret:
            assert(list(nei.iss) == list(iss))
            break

    ###########################################################################
    #### Retriever Manager
    ######################
    data_input = np.random.random((100, 2))
    data1 = np.random.random((50, 2))
    data2 = np.random.random((70, 2))

    mapper0 = None
    mapper1 = (0, 0)
    mapper2 = np.array([np.random.randint(0, 3, 100), np.zeros(100)]).T
    mapper3 = lambda idx: (np.random.randint(0, 3), 0)
    pos_mappers = [mapper0, mapper1, mapper2, mapper3]

    for mapper in pos_mappers:
        ret1 = KRetriever(data1, autolocs=data_input, info_ret=3)
        ret2 = KRetriever(data_input, info_ret=4)
        ret3 = CircRetriever(data2, info_ret=0.1, autolocs=data_input)
        gret = RetrieverManager([ret1, ret2, ret3], mapper)
        ## Test functions
        len(gret)
        gret[0]
        for neighs_info in gret:
            pass
        gret.add_retrievers(CircRetriever(data2[:], info_ret=0.1,
                                          autolocs=data_input))
        gret.add_perturbations(perturbation4)
        gret.set_selector(mapper)
        gret.retrieve_neighs(10)
        gret.retrieve_neighs(10, typeret_i=(0, 0))
        gret.retrieve_neighs([0])
#        gret.retrieve_neighs([0, 1, 2])

        #gret.compute_nets()
        gret.set_neighs_info(bool_input_idx=True)

        ## Impossible cases
        try:
            boolean = False
            gret[-1]
            boolean = True
            raise Exception("It has to halt here.")
        except:
            if boolean:
                raise Exception("It has to halt here.")
        try:
            boolean = False
            gret.add_retrievers(type(None))
            boolean = True
            raise Exception("It has to halt here.")
        except:
            if boolean:
                raise Exception("It has to halt here.")

#    ## Simple test
#    gret1 = RetrieverManager([ret1, ret2, ret3])
#    gret2 = RetrieverManager([ret1, ret2, ret3], mapper)
#    len(gret1)
#    len(gret2)
#    gret1[0]
#    gret2[0]
#    for neighs_info in gret1:
#        pass
#    for neighs_info in gret2:
#        pass
#    gret1.add_retrievers(ret3)
#    gret1.set_neighs_info(True)
#    gret2.set_neighs_info(True)
#    gret1.add_perturbations(perturbation4)
#    gret2.add_perturbations(perturbation4)
#
#    gret1.set_selector(mapper)
#    gret2.set_selector(mapper)

#    gret1.retrieve_neighs(10)
#    gret2.retrieve_neighs(10)

#    ## Impossible cases
#    try:
#        boolean = False
#        gret[-1]
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        gret.add_retrievers(type(None))
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")

#    try:
#        boolean = False
#        print 'x'*10
#        RetrieverManager([ret1, ret2, ret3], np.random.randint(0, 3, 200))
#        print 'y'*10
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
#    try:
#        boolean = False
#        print 'x'*10
#        RetrieverManager([ret1, ret2, ret3], np.random.randint(0, 5, 100))
#        print 'y'*10
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")

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

    ###########################################################################
    #### Auxiliar retriever parsing utils
    #####################################
    ## Standards inputs to tests
    #    * Retriever object
    #    * (Retriever class, main_info)
    #    * (Retriever class, main_info, pars_ret)
    #    * (Retriever class, main_info, pars_ret, autolocs)
    #
    locs = np.random.random((100, 2))
    pars_ret = {}

    retriever_info = KRetriever(locs)
    retriever_manager = _retriever_parsing_creation(retriever_info)
    retriever_info = [KRetriever(locs)]
    retriever_manager = _retriever_parsing_creation(retriever_info)
    assert(isinstance(retriever_manager, RetrieverManager))
    retriever_info = (KRetriever, locs)
    retriever_manager = _retriever_parsing_creation(retriever_info)
    assert(isinstance(retriever_manager, RetrieverManager))
    retriever_manager = _retriever_parsing_creation(retriever_info)
    assert(isinstance(retriever_manager, RetrieverManager))
    retriever_info = (KRetriever, locs, pars_ret)
    retriever_manager = _retriever_parsing_creation(retriever_info)
    assert(isinstance(retriever_manager, RetrieverManager))
    retriever_info = (KRetriever, locs, pars_ret, locs)
    retriever_manager = _retriever_parsing_creation(retriever_info)
    assert(isinstance(retriever_manager, RetrieverManager))

    retriever_manager = _retriever_parsing_creation(retriever_manager)
