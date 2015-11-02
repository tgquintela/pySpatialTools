

## Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pySpatialTools.Retrieve import GridSpatialDisc, CircRetriever
from pySpatialTools.Spatial_Relations.region_spatial_relations import regions_relation_points

## Paramters
n = 10000
ngx, ngy = 100, 100
n_types = 25
n_feats = 2

## Artificial distribution in space
locs = np.random.random((n, 2))
locs2 = np.array((locs[:, 0]*np.cos(locs[:, 1]*2*np.pi), locs[:, 0]*np.sin(locs[:, 1]*np.pi*2))).T

# Test distributions
fig1 = plt.plot(locs[:,0], locs[:, 1], '.')
fig2 = plt.plot(locs2[:,0], locs2[:, 1], '.')

# Random type
types = np.random.randint(0, 25, n)
# Random feature
feats = np.random.randn(n, n_feats)
# Join in a pandas dataframe
typevarsl = []
data1, data2 = pd.DataFrame([locs, types, feats], columns=typevarsl)

typevars = dict(zip([], typevarsl))

## Correlation computation
corrmodel = CorrelationModel(descriptmodel, aggfeatmodel)
CorrModelProcess(logfile, retriever, corrmodel, typevars)







