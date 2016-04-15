
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


from pySpatialTools.Retrieve import create_retriever_input_output

#from scipy.sparse import coo_matrix

from pySpatialTools.utils.artificial_data import \
    random_transformed_space_points, generate_random_relations_cutoffs
from pySpatialTools.Discretization import SetDiscretization
from pySpatialTools.Retrieve.aux_windowretriever import windows_iteration
from pySpatialTools.utils.perturbations import PermutationPerturbation,\
    NonePerturbation, JitterLocations, PermutationIndPerturbation,\
    ContiniousIndPerturbation, DiscreteIndPerturbation, MixedFeaturePertubation


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
    pos_boolinidx = [True, False, None]

    ## KRetriever
    pos_inforet = [2, 5, 10]
    pos_outmap = [None, _output_map]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx, pos_perturbations]
    for p in product(*pos):
        ret = KRetriever(locs, info_ret=p[0], ifdistance=p[1], input_map=p[2],
                         output_map=p[3], constant_info=p[4],
                         bool_input_idx=p[5], perturbations=p[6])
        if p[5] is False:
            i = locs[0]
        else:
            i = 0
        print i, p, ret.staticneighs, ret.neighs_info.staticneighs
        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            #neighs_info = ret[i]
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])

        ## Testing other functions and parameters
        ret.k_perturb

    ## CircRetriever
    pos_inforet = [2., 5., 10.]
    pos_outmap = [None, _output_map]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx]
    for p in product(*pos):
        ret = KRetriever(locs, info_ret=p[0], ifdistance=p[1], input_map=p[2],
                         output_map=p[3], constant_info=p[4],
                         bool_input_idx=p[5])
        if p[5] is False:
            i = locs[0]
        else:
            i = 0
        print i, p
        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            #neighs_info = ret[i]
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])

    ## SameEleRetriever
    pos_inforet = [None]
    pos_outmap = [None, _output_map]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx]
    for p in product(*pos):
        ret = SameEleNeigh(mainmapper, info_ret=p[0], ifdistance=p[1],
                           input_map=p[2], output_map=p[3], constant_info=p[4],
                           bool_input_idx=p[5])
        if p[5] is False:
            i = mainmapper.data[0]
            j = mainmapper.data[[0, 3]]
        else:
            i = 0
            j = [0, 3]
        print i, p
        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            #neighs_info = ret[i]
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            neighs_info = ret.retrieve_neighs(j, p[0])

    ## OrderEleRetriever
    pos_inforet = [pars4]
    pos_outmap = [None, _output_map]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx]
    for p in product(*pos):
        ret = OrderEleNeigh(mainmapper, info_ret=p[0], ifdistance=p[1],
                            input_map=p[2], output_map=p[3],
                            constant_info=p[4], bool_input_idx=p[5])
        if p[5] is False:
            i = mainmapper.data[0]
            j = mainmapper.data[[0, 3]]
        else:
            i = 0
            j = [0, 3]
        print i, p
        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            #neighs_info = ret[i]
            neighs_info = ret.retrieve_neighs(j)
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            neighs_info = ret.retrieve_neighs(j, p[0])

    ## LimDistanceRetriever
    pos_inforet = [pars5]
    pos_outmap = [None, _output_map]

    pos = [pos_inforet, pos_ifdistance, pos_inmap, pos_outmap,
           pos_constantinfo, pos_boolinidx]
    for p in product(*pos):
        ret = LimDistanceEleNeigh(mainmapper, info_ret=p[0], ifdistance=p[1],
                                  input_map=p[2], output_map=p[3],
                                  constant_info=p[4], bool_input_idx=p[5])
        if p[5] is False:
            i = mainmapper.data[0]
            j = mainmapper.data[[0, 3]]
        else:
            i = 0
            j = [0, 3]
        print i, p
        if p[4]:
            neighs_info = ret.retrieve_neighs(i)
            #neighs_info = ret[i]
            neighs_info = ret.retrieve_neighs(j)
        else:
            neighs_info = ret.retrieve_neighs(i, p[0])
            neighs_info = ret.retrieve_neighs(j, p[0])

#info_ret=None, autolocs=None, pars_ret=None,
#                 autoexclude=True, ifdistance=False, info_f=None,
#                 perturbations=None, relative_pos=None, input_map=None,
#                 output_map=None, constant_info=False, bool_input_idx=None,
#                 format_level=None, type_neighs=None, type_sp_rel_pos=None

    ## Implicit
    ret0 = KRetriever(locs, 3, ifdistance=True, input_map=_input_map,
                      output_map=_output_map)
    ret1 = CircRetriever(locs, 3, ifdistance=True, bool_input_idx=True)
    ret2 = KRetriever(locs1, 3, ifdistance=True, bool_input_idx=True)

    ## Explicit
    ret3 = SameEleNeigh(mainmapper, input_map=input_map,
                        bool_input_idx=True)
    ret4 = OrderEleNeigh(mainmapper, pars4, input_map=input_map,
                         bool_input_idx=True)
    ret5 = LimDistanceEleNeigh(mainmapper, pars5, input_map=input_map,
                               bool_input_idx=True)

    info_f = lambda x, y: 2
    relative_pos = lambda x, y: y
    ret6 = KRetriever(locs1, 3, ifdistance=True, constant_info=True,
                      autoexclude=False, info_f=info_f, bool_input_idx=True,
                      relative_pos=relative_pos)
    ret7 = KRetriever(locs1, 3, ifdistance=True, constant_info=True,
                      autoexclude=False, info_f=info_f, bool_input_idx=False,
                      relative_pos=relative_pos)
    ret8 = WindowsRetriever((100,), pars_ret=pars8)

    ## Retriever Manager
    gret = RetrieverManager([ret0, ret1, ret2, ret3, ret4, ret5])

    for i in xrange(n):
        ## Reduce time of computing
        if np.random.random() < 0.8:
            continue
        print 'xz'*80, i
        neighs_info = ret0.retrieve_neighs(i)
        neighs_info = ret1.retrieve_neighs(i)
        neighs_info = ret2.retrieve_neighs(i)
        neighs_info = ret3.retrieve_neighs(i)
        neighs_info = ret4.retrieve_neighs(i)
        neighs_info = ret5.retrieve_neighs(i)
        neighs_info = ret6.retrieve_neighs(i)
        neighs_info = ret7.retrieve_neighs(locs1[i])
        neighs_info = gret.retrieve_neighs(i)
        neighs_info = ret8.retrieve_neighs(i)

#    ret0._prepare_input = lambda x, kr:
#    neighs_info = ret0._retrieve_neighs_dynamic(0)

    neighs_info = ret1._retrieve_neighs_constant_nodistance(4)
    neighs_info = ret1._retrieve_neighs_constant_distance(4)
    neighs_info = ret2._retrieve_neighs_constant_nodistance(4)
    neighs_info = ret8._retrieve_neighs_constant_nodistance(8, pars8)
    neighs_info = ret8._retrieve_neighs_constant_distance(8, pars8)

    ## Retrieve-driven testing
    for idx, neighs in ret1:
            pass
    for idx, neighs in ret3:
            pass
    for idx, neighs in ret4:
            pass
    for idx, neighs in ret5:
            pass
    ret6.set_iter(2, 1000)
    for idx, neighs in ret6:
        pass

    ret8.set_iter()
#    for idx, neighs in ret8:
#        pass
#        print idx, neighs

    ## Main functions
    ret1.data_input
    ret1.data_output
    ret1.shape
    ret1[0]

    ret2.data_input
    ret2.data_output
    ret2.shape
    ret2[0]

    reindices = np.vstack([np.random.permutation(n) for i in range(5)])
    perturbation = PermutationPerturbation(reindices.T)
    ret0.add_perturbations(perturbation)

    ##
    ### TODO: __iter__
#    net = ret1.compute_neighnet()
#    net = ret2.compute_neighnet()

    ## Other external functions
    aux = np.random.randint(0, 100, 1000)
    m_in, m_out = create_retriever_input_output(aux)

    dummyret = DummyRetriever(None)
    try:
        _check_retriever(dummyret)
        raise Exception
    except:
        pass
    dummyret.retriever = None
    dummyret._default_ret_val = None
    try:
        _check_retriever(dummyret)
        raise Exception
    except:
        pass
    dummyret = DummyRetriever(None)
    dummyret._retrieve_neighs_spec = None
    dummyret._define_retriever = None
    dummyret._format_output_exclude = None
    dummyret._format_output_noexclude = None
    try:
        _check_retriever(dummyret)
        raise Exception
    except:
        pass

    ## General Retriever
    class PruebaRetriever(GeneralRetriever):
        preferable_input_idx = True
        auto_excluded = True
        constant_neighs = True
        bool_listind = True

        def __init__(self, autoexclude=True, ifdistance=False):
            bool_input_idx = True
            info_ret, info_f, constant_info = None, None, None
            self._initialization()
            self._format_output_information(autoexclude, ifdistance, None)
            self._format_exclude(bool_input_idx, self.constant_neighs)
            self._format_retriever_info(info_ret, info_f, constant_info)
            ## Format retriever function
            self._format_retriever_function()
            self._format_getters(bool_input_idx)
            self._format_preparators(bool_input_idx)
            self._format_neighs_info(bool_input_idx, 2, 'list', 'list')

        def _define_retriever(self):
            pass

        def _retrieve_neighs_general_spec(self, point_i, p, ifdistance=False,
                                          kr=0):
            return [[0]], None
    pruebaret = PruebaRetriever(True)
    pruebaret.retrieve_neighs(0)
    pruebaret.retrieve_neighs(1)
    pruebaret = PruebaRetriever(False)
    pruebaret.retrieve_neighs(0)
    pruebaret.retrieve_neighs(1)

    ### Auxiliar functions of window retriever
    def iteration_auxiliar(shape, l, center, excluded):
        matrix = np.zeros((shape)).astype(int)
        matrix = matrix.ravel()
        for inds, neighs, d in windows_iteration(shape, 1000, l, center,
                                                 excluded):
            matrix[inds] += len(neighs)
            assert(np.all(inds >= 0))
        matrix = matrix.reshape(shape)
        #import matplotlib.pyplot as plt
        #plt.imshow(matrix)
        #plt.show()
    shape, l, center, excluded = (10, 10), [4, 5], [0, 0], False
    iteration_auxiliar(shape, l, center, excluded)
    shape, l, center, excluded = (10, 10), [2, 5], [2, -1], False
    iteration_auxiliar(shape, l, center, excluded)
    shape, l, center, excluded = (10, 10), [4, 3], [-2, -1], True
    iteration_auxiliar(shape, l, center, excluded)
    shape, l, center, excluded = (10, 10), [1, 5], [2, 2], False
    iteration_auxiliar(shape, l, center, excluded)
