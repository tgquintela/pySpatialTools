
## Imports
import numpy as np
import matplotlib.pyplot as plt
from pySpatialTools.Retrieve import GridSpatialDisc, CircRetriever
from pySpatialTools.Spatial_Relations.region_spatial_relations import regions_relation_points

## Paramters
n = 10000
ngx, ngy = 100, 100

## Artificial distribution in space
locs = np.random.random((n, 2))
locs2 = np.array((locs[:, 0]*np.cos(locs[:, 1]*2*np.pi), locs[:, 0]*np.sin(locs[:, 1]*np.pi*2))).T

# Test distributions
fig1 = plt.plot(locs[:,0], locs[:, 1], '.')
fig2 = plt.plot(locs2[:,0], locs2[:, 1], '.')

## Discretization
disc = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))
regions = disc.map2id(locs)

disc2 = GridSpatialDisc((ngx, ngy), xlim=(-1, 1), ylim=(-1, 1))
regions2 = disc2.map2id(locs2)


