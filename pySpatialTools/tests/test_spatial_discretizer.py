
## Imports
import numpy as np
#import matplotlib.pyplot as plt
from pySpatialTools.Discretization import GridSpatialDisc


def test():
    ## Paramters
    n = 1000
    ngx, ngy = 100, 100

    ## Artificial distribution in space
    locs = np.random.random((n, 2))
    locs2 = np.array((locs[:, 0]*np.cos(locs[:, 1]*2*np.pi),
                      locs[:, 0]*np.sin(locs[:, 1]*np.pi*2))).T

    ## Test distributions
    #fig1 = plt.plot(locs[:,0], locs[:, 1], '.')
    #fig2 = plt.plot(locs2[:,0], locs2[:, 1], '.')

    ## Discretization
    disc = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))
    disc2 = GridSpatialDisc((ngx, ngy), xlim=(-1, 1), ylim=(-1, 1))
    regions = disc.discretize(locs)
    regions2 = disc2.discretize(locs2)
