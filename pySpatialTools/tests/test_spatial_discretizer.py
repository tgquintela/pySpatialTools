
"""
test spatial discretization
---------------------------
test for spatial discretizors.

"""

## Imports
import numpy as np
#import matplotlib.pyplot as plt
from pySpatialTools.Discretization import *
from pySpatialTools.utils.artificial_data import *

## TODO:
########
# IrregularSpatialDisc
# sklearnlike


def test():
    ## 0. Artificial data
    # Parameters
    n = 1000
    ngx, ngy = 100, 100
    # Artificial distribution in space
    fucnts = [lambda locs: locs[:, 0]*np.cos(locs[:, 1]*2*np.pi),
              lambda locs: locs[:, 0]*np.sin(locs[:, 1]*np.pi*2)]
    locs1 = random_transformed_space_points(n, 2, None)
    locs2 = random_transformed_space_points(n, 2, fucnts)
    locs3 = random_transformed_space_points(n/100, 2, None)
    # Set discretization
    elements1 = np.random.randint(0, 50, 200)
    elements2 = np.random.randint(0, 2000, 200)

    ## Test distributions
    #fig1 = plt.plot(locs[:,0], locs[:, 1], '.')
    #fig2 = plt.plot(locs2[:,0], locs2[:, 1], '.')

    ## Discretization
    # Discretization instantiation
    disc1 = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))
    disc2 = GridSpatialDisc((ngx, ngy), xlim=(-1, 1), ylim=(-1, 1))
    disc3 = BisectorSpatialDisc(locs3, np.arange(len(locs3)))
    disc4 = CircularInclusiveSpatialDisc(locs3, np.random.random(len(locs3)))
    disc5 = CircularExcludingSpatialDisc(locs3, np.random.random(len(locs3)))
    disc6 = SetDiscretization(np.random.randint(0, 2000, 50))
    disc7 = SetDiscretization(randint_sparse_matrix(0.2, (2000, 100), 1))

    # Discretization action
    regions = disc1.discretize(locs1)
    regions = disc1.discretize(locs2)
    regions = disc2.discretize(locs1)
    regions = disc2.discretize(locs2)
    regions = disc3.discretize(locs1)
    regions = disc3.discretize(locs2)
    regions = disc4.discretize(locs1)
    regions = disc4.discretize(locs2)
    regions = disc5.discretize(locs1)
    regions = disc5.discretize(locs2)
    regions = disc6.discretize(elements1)
    regions7 = disc7.discretize(elements2)

    a = randint_sparse_matrix(0.2, (2000, 100), 1)

    # Inverse discretization action

    # Contiguity
    contiguity = disc1.get_contiguity()
    contiguity = disc2.get_contiguity()
    #contiguity = disc3.get_contiguity()
    #contiguity = disc4.get_contiguity()
    #contiguity = disc5.get_contiguity()
    #contiguity = disc6.get_contiguity()
    #contiguity = disc7.get_contiguity()

    ## Other parameters and functions
    disc1.borders, disc5.borders, disc6.borders

    ## Extending coverage
    n_in, n_out = 100, 20
    relations = [np.unique(np.random.randint(0, n_out,
                                             np.random.randint(n_out)))
                 for i in range(n_in)]

    disc8 = SetDiscretization(relations)
    relations = [list(e) for e in relations]
    disc9 = SetDiscretization(relations)
    disc8.discretize(np.random.randint(0, 20, 100))
    disc9.discretize(np.random.randint(0, 20, 100))


    ## Check regions
    neighs = disc1.check_neighbors(np.array([0, 1]), 0)

    activated = disc1.get_activated_regions(locs1[:10])
    activated = disc2.get_activated_regions(locs1[:10])
    activated = disc3.get_activated_regions(locs1[:10])
    activated = disc4.get_activated_regions(locs1[:10])
    activated = disc5.get_activated_regions(locs1[:10])
    activated = disc6.get_activated_regions(elements1[:10])
    activated = disc7.get_activated_regions(elements2[:10])
    activated = disc8.get_activated_regions(np.array([0, 1]))
    activated = disc9.get_activated_regions(np.array([0, 1]))

    disc1.belong_region(locs1[0], 0)
    disc2.belong_region(locs1[1], 0)
    limits = disc1.get_limits()
    limits = disc1.get_limits(0)
    limits = disc2.get_limits()
#    limits = disc3.get_limits()
#    limits = disc4.get_limits(0)
