
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

## Spatial relations
retriever = CircRetriever(locs, True)
radis = [0.01, 0.05, 0.1, 0.25, 0.5]
n_radis = len(radis)
n_reg1, n_reg2 = np.unique(regions).shape[0], np.unique(regions2).shape[0]

relation1 = np.zeros((n_reg1, n_reg1, n_radis))
relation2 = np.zeros((n_reg2, n_reg2, n_radis))
for i in range(n_radis):
    info_ret = np.ones(n)*radis[i]

    relation1[:, :, i] = regions_relation_points(locs, regions, retriever, info_ret)
    relation2[:, :, i] = regions_relation_points(locs2, regions2, retriever, info_ret)

