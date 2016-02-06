
## Imports
import numpy as np
import matplotlib.pyplot as plt
from pySpatialTools.Retrieve.Spatial_Relations import CenterLocsRegionDistances, PointsNeighsIntersection
from pySpatialTools.Retrieve.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import KRetriever, CircRetriever
from pySpatialTools.Feature_engineering.Descriptors import Countdescriptor


def test():
    ## Paramters
    n = 1000
    ngx, ngy = 100, 100

    ## Artificial distribution in space
    locs = np.random.random((n, 2))
    locs2 = np.array((locs[:, 0]*np.cos(locs[:, 1]*2*np.pi), locs[:, 0]*np.sin(locs[:, 1]*np.pi*2))).T

    ## Test distributions
    #fig1 = plt.plot(locs[:,0], locs[:, 1], '.')
    #fig2 = plt.plot(locs2[:,0], locs2[:, 1], '.')

    ## Discretization
    disc0 = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))
    disc2 = GridSpatialDisc((ngx, ngy), xlim=(-1, 1), ylim=(-1, 1))

    # Discretizor
    centerlocs, regions_id = disc0.get_regionslocs(), disc0.get_regions_id()
    # Retrievers
    ret0 = KRetriever(centerlocs, 5, ifdistance=True, flag_auto=False, tags=regions_id)
    ret1 = CircRetriever(centerlocs, 5, ifdistance=True, flag_auto=False, tags=regions_id)


    centerlocsmetrics = CenterLocsRegionDistances(distanceorweighs=True, symmetric=True)
    sp_descriptor = (disc0, locs, KRetriever, 5, Countdescriptor)
    centerlocsmetrics.compute_distances(sp_descriptor, activated=locs)


