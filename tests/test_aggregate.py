
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

