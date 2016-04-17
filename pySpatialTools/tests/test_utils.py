
"""
test utilities
--------------
testing functions which acts as a utils.

"""

m = """In fiction, artificial intelligence is often seen a menace that allows
robots to take over and enslave humanity. While some share these concerns in
real life, a researcher suggests the robot-conquest might be more subtle than
imagined.
According to Moshe Vardi, director of the Institute for Information Technology
at Rice University in Texas, in the coming 30 years, advanced robots will
threaten tens of millions of jobs.
"""

import numpy as np
from itertools import product
from pySpatialTools.utils.artificial_data import randint_sparse_matrix
from pySpatialTools.utils.util_classes import create_mapper_vals_i,\
    Map_Vals_i
from ..utils.util_classes import Locations, SpatialElementsCollection,\
    Membership, Neighs_Info


def test():
    ## Parameters
    words = m.replace('\n', ' ').replace('.', ' ').strip().split(" ")
    ids = [hash(e) for e in words]
    functs = [lambda x: str(x)+words[i] for i in range(len(words))]

    ## Testing Elemets
    words_id = np.arange(len(words))
    words_elements = SpatialElementsCollection(words, words_id)
    words_elements2 = SpatialElementsCollection(words, list(words_id))
    words_elements = SpatialElementsCollection(words)
    ids_elements = SpatialElementsCollection(ids)
    functs_elements = SpatialElementsCollection(functs)

    # Class functions
    words_elements[0]
    try:
        words_elements[len(words_elements)]
        raise Exception
    except:
        pass
    try:
        words_elements2[words[0]]
        raise Exception
    except:
        pass
    words_elements.elements_id = None
    try:
        words_elements[words[0]]
        raise Exception
    except:
        pass

    words_elements[0]

    for e in words_elements:
        pass

    for e in words_elements2:
        pass

    words_elements == words[0]
    relabel_map = np.arange(len(words))
    try:
        words_elements.relabel_elements(range(len(words)))
    except:
        pass
    words_elements.relabel_elements(relabel_map)
    relabel_map = dict(zip(relabel_map, relabel_map))
    words_elements.relabel_elements(relabel_map)

    ids_elements[0]
    for e in ids_elements:
        pass
    ids_elements == words[0]

    functs_elements[0]
    for e in functs_elements:
        pass
    functs_elements == words[0]

    ## Locations
    locs1 = np.random.random((100, 5))
    locs2 = np.random.random((100, 1))
    locs3 = np.random.random(100)
    locs4 = np.random.random((100, 2))
    sptrans = lambda x, p: np.sin(x)

    try:
        locs = Locations(locs1, 5)
        raise Exception
    except:
        pass
    try:
        locs = Locations(locs1, list(range(len(locs1)+1)))
        raise Exception
    except:
        pass
    try:
        tags = list(range(len(locs1)))
        tags[0] = 1
        locs = Locations(locs1, tags)
        raise Exception
    except:
        pass

    locs = Locations(locs1)
    locsbis = Locations(locs1, list(range(len(locs1))))
    try:
        locsbis[-1]
        raise Exception
    except:
        pass
    locsbis[0]
    locs[0]
    assert((locs == locs1[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})
    locs._check_coord(0)
    locs._check_coord(locs[0])
    locs._check_coord([0, 3])
    locs._check_coord([locs1[0], locs1[3]])
    locs.in_radio(locs[0], 0.2)

    locs = Locations(locs2)
    assert((locs == locs2[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})
    locs.in_manhattan_d(locs[0], 0.2)

    locs = Locations(locs3)
    assert((locs == locs3[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})

    locs = Locations(locs4)
    locs.in_block_distance_d(np.random.random((1, 2)), 0.2)

    ## Membership
    n_in, n_out = 100, 20
    relations = [np.unique(np.random.randint(0, n_out,
                                             np.random.randint(n_out)))
                 for i in range(n_in)]
    relations = [list(e) for e in relations]
    memb1 = Membership(relations)

    memb1.to_network()
    memb1.to_dict()
    memb1.to_sparse()
    memb1.reverse_mapping()
    memb1.getcollection(0)
    memb1.collections_id
    memb1.n_collections
    memb1.n_elements
    memb1.membership
    str(memb1)
    memb1[0]
    memb1 == 0
    for e in memb1:
        pass

#    op2 = np.all([t == dict for t in types])
    relations = [dict(zip(e, len(e)*[{'membership': 1}])) for e in relations]
    memb1_dict = Membership(relations)
    memb1_dict.to_network()
    memb1_dict.to_dict()
    memb1_dict.to_sparse()
    memb1_dict.reverse_mapping()
    memb1_dict.getcollection(0)
    memb1_dict.collections_id
    memb1_dict.n_collections
    memb1_dict.n_elements
    memb1_dict.membership

    memb2 = Membership(np.random.randint(0, 20, 100))
    memb2.to_network()
    memb2.to_dict()
    memb2.to_sparse()
    memb2.reverse_mapping()
    memb2.getcollection(0)
    memb2.collections_id
    memb2.n_collections
    memb2.n_elements
    memb2.membership
    str(memb2)
    memb2[0]
    memb2 == 0
    for e in memb2:
        pass

    sparse = randint_sparse_matrix(0.2, (200, 100), 1)
    memb3 = Membership(sparse)
    memb3.to_dict()
    memb3.to_network()
    memb3.to_sparse()
    memb3.reverse_mapping()
    memb3.getcollection(0)
    memb3.collections_id
    memb3.n_collections
    memb3.n_elements
    memb3.membership
    str(memb3)
    memb3[0]
    memb3 == 0
    for e in memb3:
        pass

    relations = [[np.random.randint(10)] for i in range(50)]
    memb4 = Membership(relations)
    memb4.to_network()
    memb4.to_dict()
    memb4.to_sparse()
    memb4.reverse_mapping()
    memb4.getcollection(0)
    memb4.collections_id
    memb4.n_collections
    memb4.n_elements
    memb4.membership
    str(memb4)
    memb4[0]
    memb4 == 0
    for e in memb4:
        pass

    relations[0].append(0)
    memb5 = Membership(relations)
    memb5.to_network()
    memb5.to_dict()
    memb5.to_sparse()
    memb5.reverse_mapping()
    memb5.getcollection(0)
    memb5.collections_id
    memb5.n_collections
    memb5.n_elements
    memb5.membership
    str(memb5)
    memb5[0]
    memb5 == 0
    for e in memb5:
        pass

    relations[0].append(0)
    memb6 = Membership((sparse, np.arange(100)))
    memb6.to_network()
    memb6.to_dict()
    memb6.to_sparse()
    memb6.reverse_mapping()
    memb6.getcollection(0)
    memb6.collections_id
    memb6.n_collections
    memb6.n_elements
    memb6.membership
    str(memb6)
    memb6[0]
    memb6 == 0
    for e in memb6:
        pass

    ## Mapper vals
    feat_arr0 = np.random.randint(0, 20, 100)

    def map_vals_i_t(s, i, k):
        k_p, k_i = s.features[0]._map_perturb(k)
        i_n = s.features[0]._perturbators[k_p].apply2indice(i, k_i)
        return feat_arr0[i_n].ravel()[0]
    map_vals_i = create_mapper_vals_i(map_vals_i_t, feat_arr0)

    # correlation
    map_vals_i = create_mapper_vals_i('correlation', feat_arr0)
    map_vals_i = create_mapper_vals_i(('correlation', 100, 20), feat_arr0)
    map_vals_i = create_mapper_vals_i('matrix')
    map_vals_i = create_mapper_vals_i('matrix', feat_arr0)
    map_vals_i = create_mapper_vals_i(('matrix', 20), list(feat_arr0))
    map_vals_i = create_mapper_vals_i(('matrix', 100, 20), len(feat_arr0))
    map_vals_i = create_mapper_vals_i('matrix', slice(0, 100, 1))
    map_vals_i.set_prefilter(slice(0, 100, 1))
    map_vals_i.set_prefilter(10)
    map_vals_i.set_prefilter([0, 2])
    map_vals_i.set_sptype('correlation')
    map_vals_i[(None, [0], 0)]

    map_vals_i = create_mapper_vals_i(map_vals_i)
    map_vals_i = create_mapper_vals_i(feat_arr0.reshape(100, 1))
    map_vals_i = create_mapper_vals_i(None)

    map_vals_i = Map_Vals_i(100)
    map_vals_i = Map_Vals_i((1000, 20))
    map_vals_i = Map_Vals_i(map_vals_i)
    map_vals_i = Map_Vals_i(memb1)

    ###########################################################################
    ############################### Neighs_Info ###############################

    ### Creation of possible combinations
#    pos_format_set_info = [None, 'integer', 'list', 'list_only', 'list_list',
#                           'list_tuple', 'list_tuple1', 'list_tuple2',
#                           'array', 'slice', 'tuple', 'tuple_int',
#                           'tuple_slice', 'tuple_tuple', 'tuple_others']
    pos_ifdistance = [True, False, None]
    pos_constant_neighs = [True, False, None]
    pos_format_get_k_info = [None, "general", "default", "list", "integer"]
    pos_format_get_info = [None, "default", "general"]
    pos_type_neighs = [None, 'general', 'array', 'list', 'slice']
    pos_type_sp_rel_pos = [None, 'general', 'array', 'list']
    pos_format_level = [None, 0, 1, 2, 3]
    pos_format_structure = [None, 'raw', 'tuple', 'tuple_only', 'tuple_tuple',
                            'list_tuple_only', 'tuple_list_tuple']
    pos_staticneighs = [None, True, False]

    pos = [pos_constant_neighs, pos_ifdistance, pos_format_get_info,
           pos_format_get_k_info, pos_format_structure, pos_format_level,
           pos_type_neighs, pos_type_sp_rel_pos, pos_staticneighs]
    ###############################

    ### Creation of possible inputs
    creator_lvl = lambda lvl: tuple(np.random.randint(1, 10, lvl))
    creator2_lvl = lambda sh: tuple(list(sh) + [np.random.randint(5)])

    extend_list = lambda lista, n: [lista for i in range(n)]
    extend_array = lambda array, n: np.array([array for i in range(n)])

    neighs0 = lambda: np.random.randint(100)
    sp_rel_pos0 = lambda: np.random.random()

    def create_neighs(sh, type_):
        neighs = neighs0()
        for i in range(len(sh)):
            if type_ == 'array':
                neighs = extend_array(neighs, sh[len(sh)-i-1])
            else:
                neighs = extend_list(neighs, sh[len(sh)-i-1])
#        if type(neighs) == int:
#            if type_ == 'array':
#                neighs = np.array([neighs])
#            else:
#                neighs = [neighs]
        return neighs

    def create_sp_rel_pos(sh, type_):
        sp_rel_pos = np.array([sp_rel_pos0()
                               for i in range(np.random.randint(1, 4))])
        for i in range(len(sh)):
            if type_ == 'array':
                sp_rel_pos = extend_array(sp_rel_pos, sh[len(sh)-i-1])
            else:
                sp_rel_pos = extend_list(sp_rel_pos, sh[len(sh)-i-1])
        if len(sh) == 0:
            if type_ == 'list':
                sp_rel_pos = list(sp_rel_pos)
        return sp_rel_pos

#    neighs_int = lambda: np.random.randint(100)
#    neighs_list = lambda x: list([neighs_int() for i in range(x)])
#    neighs_array = lambda x: np.array([neighs_int() for i in range(x)])
    neighs_slice = lambda top: slice(0, top)
    ###############################
    level_dependent = ['list_tuple_only', 'tuple_list_tuple']

    ### Testing possible combinations
    k = 0
    for p in product(*pos):
        ## General instantiation. It has to be able to eat any input
        neighs_info_general = Neighs_Info()
        ## Defintiion of forbidden combinations
        bool_error = p[4] in level_dependent and p[5] != 2
        bool_error = bool_error or p[2] == "default" or p[3] == "default"
        ## Testing raising errors of forbidden combinations:
        if bool_error:
            try:
                boolean = False
                neighs_info = Neighs_Info(constant_neighs=p[0],
                                          ifdistance=p[1],
                                          format_get_info=p[2],
                                          format_get_k_info=p[3],
                                          format_structure=p[4],
                                          format_level=p[5],
                                          type_neighs=p[6],
                                          type_sp_rel_pos=p[7],
                                          staticneighs=p[8])
                boolean = True
            except:
                if boolean:
                    raise Exception("It has to halt here.")
            continue

        if p[4] == 'list_tuple_only' and p[0]:
            continue
        ## TESTING:
        if p[3] == 'integer':
            continue
#        print p

        ## Instantiation
        neighs_info = Neighs_Info(constant_neighs=p[0], ifdistance=p[1],
                                  format_get_info=p[2], format_get_k_info=p[3],
                                  format_structure=p[4], format_level=p[5],
                                  type_neighs=p[6], type_sp_rel_pos=p[7],
                                  staticneighs=p[8])
        neighs_info.set_information(10, 100)

        ## Presetting
        lvl = np.random.random(4) if p[5] is None else p[5]
        sh = creator_lvl(lvl)
        k_len = sh[0] if len(sh) == 3 else np.random.randint(1, 9)
        iss_len = sh[len(sh)-2] if len(sh) > 1 else np.random.randint(1, 100)
        nei_len = sh[-1] if len(sh) > 0 else np.random.randint(100)
        sh_static = sh
        if len(sh_static) == 3:
            sh_static = list(sh_static)
            sh_static[0] = k_len
            sh_static = tuple(sh_static)
#        if p[0] is True:
#            basic_sh = [] if len(sh) == 0 else [1]
#            sh_static = tuple(basic_sh+[sh[i] for i in range(1, len(sh))])
        # Neighs creation
        if p[6] == 'slice':
            neighs = neighs_slice(nei_len)
        else:
            # Use type_neighs and level
            neighs = create_neighs(sh_static, p[6])
        # Sp_rel_pos
        sp_rel_pos = create_sp_rel_pos(sh_static, p[7])
        if p[7] == 'list':
#            print type(sp_rel_pos), p, sh
            assert(type(sp_rel_pos) == list)

        ## Create structure p[4], p[6], p[7] None
        # Create structure
        if p[4] == 'raw':
            neighs_nfo = neighs
        elif p[4] == 'tuple':
            neighs_nfo = (neighs, np.arange(k_len))
        elif p[4] == 'tuple_only':
            neighs_nfo = (neighs, sp_rel_pos)
        elif p[4] == 'tuple_tuple':
            neighs_nfo = ((neighs, sp_rel_pos), np.arange(k_len))
        elif p[4] == 'list_tuple_only':
            neighs_nfo = [(neighs, sp_rel_pos) for i in range(k_len)]
        elif p[4] == 'tuple_list_tuple':
            neighs_nfo = ([(neighs, sp_rel_pos) for i in range(k_len)],
                          range(k_len))
        else:
            neighs_nfo = neighs

        tupletypes = ['tuple', 'tuple_only', 'tuple_tuple', 'list_tuple_only',
                      'tuple_list_tuple']
        if p[6] == 'slice' and p[4] in tupletypes:
            continue

#        print 'neighs_info', k, neighs_nfo, type(neighs_nfo), p
        neighs_info.set(neighs_nfo, range(iss_len))
        ks = [0] if neighs_info.ks is None else neighs_info.ks
        neighs_info.get_information(ks)
        ## Special cases
        # Slice type
        if p[6] == 'slice' and p[4] == 'raw':
            neighs_info.set(5, range(iss_len))
            neighs_info.set((2, 6), range(iss_len))
            neighs_info.set(None, range(iss_len))

        # Important general functions
        neighs_info.any()
        neighs_info.empty()
        neighs_info.shape
        neighs_info._get_neighs_general()
        neighs_info._general_get_rel_pos()
        neighs_info.reset_functions()
        if p[6] == 'slice':
            neighs_info._get_neighs_general()
        # Reset structure
        if p[4] not in ['list_tuple_only', 'tuple_list_tuple']:
            # Rewrite level problems avoiding
            neighs_info.reset_level(p[5])
            neighs_info.reset_structure(p[4])
        neighs_info_general.set(neighs_nfo, range(iss_len))

        k += 1
#        print '-'*20, k

    #* integer {neighs}
    neighs_info = Neighs_Info()
    neighs_info.shape
    try:
        boolean = False
        neighs_info._default_get_neighs()
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        neighs_info._default_get_information()
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    neighs_info._format_set_iss('int')
    neighs_info._format_set_iss('list')
    neighs_info._format_set_iss('null')
    neighs_info = Neighs_Info()
    neighs_info.set_information(10, 10)
    neighs_info.set(5)
    neighs_info.set(([0], [5.]))
    #* list of integers {neighs}
    neighs_info.reset()
    neighs_info.set_information(10, 10)
    neighs_info.set([[0, 4]])
    try:
        boolean = False
        neighs_info._kret = 10
        neighs_info._integer_get_k(100)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    neighs_info._get_neighs_list_static()
    #* list of lists of integers {neighs for some iss}
    neighs_info.reset()
    neighs_info.set([[0, 4], [0, 3]])
    #* list of lists of lists of integers {neighs for some iss and ks}
    neighs_info.reset()
    neighs_info.set([[[0, 4], [0, 3]]])
    neighs_info.staticneighs = False
    neighs_info._array_only_set_rel_pos(np.array([[2], [3]]))
    neighs_info._general_set_iss(True)
    neighs_info._set_rel_pos_general_array(np.array([[2], [3]]))
    neighs_info._set_rel_pos_general_array(np.array([[[[2], [3]]]]))
    neighs_info._list_list_only_set_rel_pos(np.array([[[5]]]))
    neighs_info.staticneighs = True
    neighs_info._general_set_iss(True)
    neighs_info._set_rel_pos_general_list([[[[2], [3]]]])
    neighs_info._general_set_rel_pos(None)

    try:
        boolean = False
        neighs_info._general_set_rel_pos(True)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    #* numpy array 1d, 2d, 3d {neighs}
    neighs_info = Neighs_Info()
    neighs_info._set_neighs_general_array(np.array(5))
    neighs_info._set_neighs_slice(None)
    neighs_info.reset()

    try:
        boolean = False
        neighs_info._general_set_neighs(True)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")

    neighs_info.reset()
    neighs_info.set(np.array([[[0, 4], [0, 3]]]))
    neighs_info.staticneighs = False
    neighs_info._set_rel_pos_dim(5)
    neighs_info.reset()
    neighs_info.set(np.array([[0, 4], [0, 3]]))
    neighs_info.reset()
    neighs_info.set(np.array([0, 4]))
    #* tuple of neighs

    # Empty cases
    neighs_info.reset()
    neighs_info.set([[]])
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info._general_set_iss(0)
    neighs_info._general_set_iss([0])
    neighs_info._general_set_iss(np.array([0]))
    neighs_info.staticneighs = True
    neighs_info._general_set_iss(np.array([0]))
    neighs_info._set_rel_pos_general_list(np.array([0]))
    neighs_info._set_rel_pos_general_list(np.array([[0]]))
    neighs_info._set_rel_pos_general_list(np.array([[]]))

    ## Get k
    neighs_info._integer_get_k([4])
    neighs_info._integer_get_k(5)
    neighs_info._default_get_k()
    try:
        boolean = False
        neighs_info._integer_get_k(100000)
        boolean = True
    except:
        if boolean:
            raise Exception("It has to halt here.")
    neighs_info._general_get_k()
    neighs_info._general_get_k(4)
    neighs_info._general_get_k(range(2))

    ## Set iss
    neighs_info._general_set_iss(range(3))
    neighs_info._general_set_iss(5)
    neighs_info._int_set_iss(8)
    neighs_info._list_set_iss(range(3))
    neighs_info._null_set_iss(5)

    neighs_info.reset()
    neighs_info.set(([[]], [[]]))
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info.set(([[]], [[]]), [0])
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info.set(([np.array([])], [np.array([])]), [0])
    assert(neighs_info.empty())

    neighs_info = Neighs_Info(format_structure='tuple_tuple',
                              type_neighs='list', type_sp_rel_pos='list',
                              format_level=2)
    neighs_info.set((([np.array([])], [np.array([[]])]), [0]))
    neighs, _, _, _ = neighs_info.get_information(0)

    ## Empty assert
    neighs_info = Neighs_Info()
    neighs_info.set((([0], [0]), 0))
    assert(not neighs_info.empty())
    for a, b, c, d in neighs_info:
        pass
