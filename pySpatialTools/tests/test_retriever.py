
"""
test retrievers
---------------
test for retrievers precoded and framework of retrievers.

"""

import numpy as np

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

    _input_map = lambda s, i: i
    _output_map = [lambda s, i, x: x]
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

        def __init__(self, autoexclude=True, ifdistance=False):
            self._initialization()
            self._format_output_information(autoexclude, ifdistance, None)
            self._format_exclude(True)
            self._format_retriever_info(None, None, None)
            ## Format retriever function
            self._format_retriever_function(True)
            self._format_preparators(True)
            self._format_neighs_info(True, 2, 'list', 'list')

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
