
"""
Testing sampling
----------------
testing functions which helps in spatial sampling

"""

#import networkx as nx
#from scipy.sparse import coo_matrix
#from pySpatialTools.utils.artificial_data import\
#    generate_random_relations_cutoffs
import numpy as np
from pySpatialTools.Sampling.sampling_from_space import *
from pySpatialTools.Sampling.sampling_from_points import *
from pySpatialTools.Sampling.auxiliary_functions import *
from pySpatialTools.Discretization.Discretization_2d import GridSpatialDisc


def test():
#    ## Generate sp_relations
#    sp_relations = generate_random_relations_cutoffs(100)
#    connected = nx.connected_components(sp_relations.relations)
    ###########################################################################
    ###########################################################################
    ############################## Test sampling ##############################
    ###########################################################################
    ## Parameters
    n, n_e = 100, 1000
    ngx, ngy = 100, 100
    limits = np.array([[0.1, -0.1], [0.5, 0.6]])

    disc = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))

    p_cats = np.random.randint(0, 10, n_e)
    locs = np.random.random((n_e, 2))

    region_weighs = np.random.random(ngx*ngy)
    point_weighs = np.random.random(n_e)

    #### Functions to test
    ######################
    ### Sampling core
    weighted_sampling_with_repetition(p_cats, n)
    weighted_sampling_with_repetition(p_cats, n, point_weighs)
    weighted_sampling_without_repetition(p_cats, n)
    weighted_sampling_without_repetition(p_cats, n, point_weighs)
    weighted_nonfixed_sampling_without_repetition(p_cats, n)
    weighted_nonfixed_sampling_without_repetition(p_cats, n, point_weighs)

    ### Sampling from region
    uniform_points_sampling(limits, n)
    weighted_region_space_sampling(disc, n, point_weighs)

    ## Sampling from points
    weighted_point_sampling(locs, n)
    weighted_point_sampling(locs, n, point_weighs)
    uniform_points_points_sampling(limits, locs, n)
    weighted_region_points_sampling(disc, locs, n)
    weighted_region_points_sampling(disc, locs, n, region_weighs)

