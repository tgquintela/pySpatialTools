
"""
Testing Neighs_Info
-------------------
Neighs_Info is the main class which contains the neighbourhood information and
it links to some db.

"""

import numpy as np
from itertools import product
from ..utils.util_classes import Neighs_Info
from ..utils.util_classes.neighs_info import *


def test():
    import warnings
    warnings.simplefilter("ignore")
    ###########################################################################
    ############################### Neighs_Info ###############################

    ### Creation of possible combinations
#    pos_format_set_info = [None, 'integer', 'list', 'list_only', 'list_list',
#                           'list_tuple', 'list_tuple1', 'list_tuple2',
#                           'array', 'slice', 'tuple', 'tuple_int',
#                           'tuple_slice', 'tuple_tuple', 'tuple_others']

    ############################# USEFUL FUNCTIONS ############################
    joinpos = lambda x, y: x

    ### Creation of possible inputs
    creator_lvl = lambda lvl: tuple(np.random.randint(1, 10, int(lvl)))
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

    def creation_neighs_nfo(p, sh, k_len, nei_len):
        ## Presetting
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
        return neighs_nfo

    ###########################################################################

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
    ###############################
    level_dependent = ['list_tuple_only', 'tuple_list_tuple']

    ### Testing possible combinations
    k = 0
    for p in product(*pos):
        ## General instantiation. It has to be able to eat any input
        neighs_info_general = Neighs_Info()
        ## Defintiion of forbidden combinations
        bool_error = p[4] in level_dependent and p[5] != 2
#        bool_error = bool_error or p[2] == "default" or p[3] == "default"
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
                raise Exception("It has to halt here.")
            except:
                if boolean:
                    raise Exception("It has to halt here.")
            continue
        ## Avoid non-allowed
        if p[2] == "default" or p[3] == "default":
            continue
        if p[4] == 'list_tuple_only' and p[0]:
            continue
        tupletypes = ['tuple', 'tuple_only', 'tuple_tuple', 'list_tuple_only',
                      'tuple_list_tuple']
        if p[6] == 'slice' and p[4] in tupletypes:
            continue
        ## TESTING:
        if p[3] == 'integer':
            continue
#        print p
        ## Save effort
        if p[4] == 'tuple':
            if np.random.random() < 0.9:
                continue

        ## Instantiation
        neighs_info = Neighs_Info(constant_neighs=p[0], ifdistance=p[1],
                                  format_get_info=p[2], format_get_k_info=p[3],
                                  format_structure=p[4], format_level=p[5],
                                  type_neighs=p[6], type_sp_rel_pos=p[7],
                                  staticneighs=p[8])
        neighs_info.set_information(100, 100)

        lvl = np.random.randint(4) if p[5] is None else p[5]
        sh = creator_lvl(lvl)
        iss_len = sh[len(sh)-2] if len(sh) > 1 else np.random.randint(1, 100)
        k_len = sh[0] if len(sh) == 3 else np.random.randint(1, 9)
        nei_len = sh[-1] if len(sh) > 0 else np.random.randint(100)
        neighs_nfo = creation_neighs_nfo(p, sh, k_len, nei_len)

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

        ## Testing joining
        logi = neighs_info.ifdistance and neighs_info.sp_relative_pos is None
        if np.random.random() < 0.1:
            if logi or type(neighs_info.idxs) == slice:
                try:
                    boolean = False
                    mode = ['and', 'or', 'xor'][np.random.randint(0, 3)]
                    neighs_info.join_neighs(neighs_info, mode, joinpos)
                    boolean = True
                    raise Exception("It has to halt here.")
                except:
                    if boolean:
                        raise Exception("It has to halt here.")
            else:
                neighs_info.join_neighs(neighs_info, 'and', joinpos)
                neighs_info.join_neighs(neighs_info, 'or', joinpos)
                neighs_info.join_neighs(neighs_info, 'xor', joinpos)
                join_neighsinfo_AND_general(neighs_info, neighs_info, joinpos)
                join_neighsinfo_OR_general(neighs_info, neighs_info, joinpos)
                join_neighsinfo_XOR_general(neighs_info, neighs_info, joinpos)
                neighs_nfo2 = creation_neighs_nfo(p, sh, k_len, nei_len)
                neighs_info2 = neighs_info.copy()
                neighs_info2.set(neighs_nfo2, range(iss_len))
                neighs_info2.sp_relative_pos = neighs_info.sp_relative_pos
                neighs_info.join_neighs(neighs_info2, 'and', joinpos)
                neighs_info.join_neighs(neighs_info2, 'or', joinpos)
                neighs_info.join_neighs(neighs_info2, 'xor', joinpos)
                join_neighsinfo_AND_general(neighs_info, neighs_info2, joinpos)
                join_neighsinfo_OR_general(neighs_info, neighs_info2, joinpos)
                join_neighsinfo_XOR_general(neighs_info, neighs_info2, joinpos)

        k += 1
#        print '-'*20, k

    #* integer {neighs}
    neighs_info = Neighs_Info()
    neighs_info.shape
    try:
        boolean = False
        neighs_info._default_get_neighs()
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        neighs_info._default_get_information()
        boolean = True
        raise Exception("It has to halt here.")
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
        neighs_info._integer_get_k(100000)
        boolean = True
        raise Exception("It has to halt here.")
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
        raise Exception("It has to halt here.")
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
        raise Exception("It has to halt here.")
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
    neighs_info = Neighs_Info(staticneighs=True)
    neighs_info.set([[]])

    neighs_info.reset()
    try:
        boolean = False
        neighs_info._set_tuple_only_structure((range(2), None, 5))
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    neighs_info._set_structure_list([np.array(range(10))])
    neighs_info.staticneighs = True
    # Testing setting rel_pos strange cases
    neighs_info._general_set_rel_pos(5)
    neighs_info._set_rel_pos_general_list(np.array([0]))
    neighs_info._set_rel_pos_general_list(np.array([[0]]))
    neighs_info._set_rel_pos_general_list(np.array([]))
    neighs_info._set_rel_pos_general_list(np.array([[]]))
    neighs_info._set_rel_pos_general_list(np.array([[[]]]))
    neighs_info._set_rel_pos_number(5)
    neighs_info.level = 0
    neighs_info._set_rel_pos_general_list(np.array([0, 3]))

    ## Get k
    neighs_info._kret = 10
    neighs_info._integer_get_k([4])
    neighs_info._integer_get_k(5)
    neighs_info._default_get_k()
    try:
        boolean = False
        neighs_info._integer_get_k(100000)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    neighs_info._general_get_k()
    neighs_info._general_get_k(4)
    neighs_info._general_get_k(range(2))

    ## Set iss
    neighs_info._general_set_iss(np.array([0]))
    neighs_info._general_set_iss(np.array([0]))
    neighs_info._general_set_iss(0)
    neighs_info._general_set_iss([0])
    neighs_info._general_set_iss(range(3))
    neighs_info._general_set_iss(5)
    neighs_info._int_set_iss(8)
    neighs_info._list_set_iss(range(3))
    neighs_info._null_set_iss(5)

    ## Stress empty tests
    neighs_info.reset()
    neighs_info.set(([[]], [[]]))
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info.set(([[], []]))
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info.set([])
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info.set(([]))
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info.set(([[]], [[]]), [0])
    assert(neighs_info.empty())

    neighs_info.reset()
    neighs_info.set(([np.array([])], [np.array([])]), [0])
    assert(neighs_info.empty())

    neighs_info = Neighs_Info(staticneighs=True)
    neighs_info._set_neighs_general_array(np.array(5))

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

    ## Check proper outputs
    # Neighs information
    neighs_info = Neighs_Info()
    sh = creator_lvl(3)
    neighs = np.random.randint(0, 20, np.prod(sh)).reshape(sh)
    sp_relative_pos = np.random.random((tuple(list(sh)+[3])))
    ks = range(sh[0])
    iss = range(sh[1])

    neighs_info.set((neighs, sp_relative_pos), iss)
    neighs_info.check_output_standards(neighs, sp_relative_pos, ks, iss)

    # Check wrong cases
    try:
        boolean = False
        sh1 = list(sh)
        sh1[0] = sh[0]+1
        sh1 = tuple(sh1)
        neighs = np.random.randint(0, 20, np.prod(sh1)).reshape(sh1)
        sp_relative_pos = np.random.random((tuple(list(sh1)+[3])))
        neighs_info.check_output_standards(neighs, sp_relative_pos, ks, iss)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        sh1 = list(sh)
        sh1[1] = sh[1]+1
        sh1 = tuple(sh1)
        neighs = np.random.randint(0, 20, np.prod(sh1)).reshape(sh1)
        sp_relative_pos = np.random.random((tuple(list(sh1)+[3])))
        neighs_info.check_output_standards(neighs, sp_relative_pos, ks, iss)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        neighs_info.check_output_standards(None, sp_relative_pos, ks, iss)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
    try:
        boolean = False
        neighs_info.idxs = None
        neighs_info.assert_goodness()
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("It has to halt here.")
# Why should be halt?
#    try:
#        boolean = False
#        neighs_info.idxs = [[]]
#        neighs_info.staticneighs = True
#        neighs_info.assert_goodness()
#        boolean = True
#        raise Exception("It has to halt here.")
#    except:
#        if boolean:
#            raise Exception("It has to halt here.")
