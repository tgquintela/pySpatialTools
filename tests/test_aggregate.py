
## Tested properly

import numpy as np
import pandas as pd
from pySpatialTools.Retrieve.spatialdiscretizer import GridSpatialDisc
from pySpatialTools.Preprocess import Aggregator


locs = np.random.random((10000, 2))


#discretizor = GridSpatialDisc(grid_size=(100,100), xlim=(0, 1), ylim=(0, 1))
#Neigh = GridNeigh(locs, (100,100), (0, 1), (0, 1))
#locs_grid = Neigh.apply_grid(locs)

typevars = {'loc_vars':['x', 'y'], 'feat_vars': ['a'], 'agg_var': 'b'}
agg = Aggregator(typevars=typevars)

dflocs = pd.DataFrame(locs, columns=['x', 'y'])
dftype = pd.DataFrame(np.random.randint(0, 20,10000), columns=['a'])
dfagg = pd.DataFrame(np.random.randint(0, 100,10000), columns=['b'])
df = pd.concat([dflocs, dftype, dfagg], axis=1)

reindices = np.zeros((10000, 10+1)).astype(int)
reindices[:, 0] = np.arange(10000)
for i in range(1, 11):
    reindices[:, i] = np.random.permutation(np.arange(10000).astype(int))

res1, res2 = agg.retrieve_aggregation(df, reindices)




def test():
    ## Artificial distribution in space
    locs = np.random.random((n, 2))
    locs2 = np.array((locs[:, 0]*np.cos(locs[:, 1]*2*np.pi),
                      locs[:, 0]*np.sin(locs[:, 1]*np.pi*2))).T

    ## Test distributions
    #fig1 = plt.plot(locs[:,0], locs[:, 1], '.')
    #fig2 = plt.plot(locs2[:,0], locs2[:, 1], '.')

    ## Discretization
    ngx, ngy = 100, 100
    disc0 = np.random.randint(0, 10, n)
    disc1 = GridSpatialDisc((ngx, ngy), xlim=(0, 1), ylim=(0, 1))
    disc2 = GridSpatialDisc((ngx, ngy), xlim=(-1, 1), ylim=(-1, 1))

    n = 10000
    locs = np.random.random((n, 2))*100
    feat_arr0 = np.random.randint(0, 20, (n, 1))
    feat_arr1 = np.random.random((n, 10))
    reindices = np.arange(n).reshape((n, 1))

    avgdesc = AvgDescriptor(feat_arr1)
    countdesc = Countdescriptor(feat_arr0)

    sp_descriptor = (disc0, locs, KRetriever, 5, Countdescriptor)
    sp_descriptor = create_sp_descriptor_points_regs(sp_descriptor, regions_id, elements_i)
    sp_descriptor.reindices = np.arange(locs.shape[0]).reshape((locs.shape[0], 1))
    net = sp_descriptor.compute_net()[:, :, 0]
    countdesc.compute_aggdescriptors(discretizor, regionretriever, locs)

    sp_descriptor = (disc1, locs, KRetriever, 5, Countdescriptor)
    sp_descriptor = create_sp_descriptor_points_regs(sp_descriptor, regions_id, elements_i)
    sp_descriptor.reindices = np.arange(locs.shape[0]).reshape((locs.shape[0], 1))
    net = sp_descriptor.compute_net()[:, :, 0]
    countdesc.compute_aggdescriptors(discretizor, regionretriever, locs)

    sp_descriptor = (disc2, locs, KRetriever, 5, Countdescriptor)
    sp_descriptor = create_sp_descriptor_points_regs(sp_descriptor, regions_id, elements_i)
    sp_descriptor.reindices = np.arange(locs.shape[0]).reshape((locs.shape[0], 1))
    net = sp_descriptor.compute_net()[:, :, 0]
    countdesc.compute_aggdescriptors(discretizor, regionretriever, locs)


    avgdesc = AvgDescriptor(feat_arr1)
    countdesc = Countdescriptor(feat_arr0)

    avgdesc.compute_aggdescriptors(discretizor, regionretriever, locs)
