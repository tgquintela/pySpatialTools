
"""
test retrievers
---------------
test for retrievers precoded and framework of retrievers.

"""

import numpy as np

from pySpatialTools.Retrieve import KRetriever, CircRetriever,\
    RetrieverManager, SameEleNeigh, OrderEleNeigh, LimDistanceEleNeigh

from pySpatialTools.Retrieve import create_retriever_input_output

#from scipy.sparse import coo_matrix

from pySpatialTools.utils.artificial_data import \
    random_transformed_space_points, generate_random_relations_cutoffs
from pySpatialTools.Discretization import SetDiscretization


def test():
    ## Parameters
    n = 1000
    # Implicit
    locs = np.random.random((n, 2))*100
    locs1 = random_transformed_space_points(n, 2, None)*10
    # Explicit
    disc0 = SetDiscretization(np.random.randint(0, 20, 1000))
    input_map = lambda s, x: disc0.discretize(x)
    pars4 = {'order': 4}
    pars5 = {'lim_distance': 2}
    mainmapper = generate_random_relations_cutoffs(20, store='sparse')

    ## Implicit
    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, 3, ifdistance=True)
    ret2 = KRetriever(locs1, 3, ifdistance=True)

    ## Explicit
    ret3 = SameEleNeigh(mainmapper, input_map=input_map)
    ret4 = OrderEleNeigh(mainmapper, pars4, input_map=input_map)
    ret5 = LimDistanceEleNeigh(mainmapper, pars5, input_map=input_map)

    ## Retriever Manager
    gret = RetrieverManager([ret0, ret1, ret2, ret3, ret4, ret5])

    for i in xrange(n):
        ## Reduce time of computing
        if np.random.random() < 0.8:
            continue
        neighs_info = ret0.retrieve_neighs(i)
        neighs_info = ret1.retrieve_neighs(i)
        neighs_info = ret2.retrieve_neighs(i)
        neighs_info = ret3.retrieve_neighs(i)
        neighs_info = ret4.retrieve_neighs(i)
        neighs_info = ret5.retrieve_neighs(i)
        neighs_info = gret.retrieve_neighs(i)

    ## Main functions
    ret1.data_input
    ret1.data_output
    ret1.shape
    ret1[0]

    ret2.data_input
    ret2.data_output
    ret2.shape
    ret2[0]

    net = ret1.compute_neighnet()
    net = ret2.compute_neighnet()

    ## Other external functions
    m_in, m_out = create_retriever_input_output(np.random.randint(0, 100, 1000))
