
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

# Random type
types = np.random.randint(0, 25, n)
# Random feature
feat_arr = np.random.randn(n, n_feats)
# Join in a pandas dataframe
typevarsl = []
data1, data2 = pd.DataFrame([locs, types, feat_arr], columns=typevarsl)

## Corr Computation
corrmodelprocess = CorrModelProcess(logfile, retriever, corrmodel, typevars)
corr_loc = corrmodelprocess.compute_corr(data, reindices=None)

## Corr Recommendation
recommender = PjensenRecommender()
q = recommender.compute_quality(corr_loc, count_matrix, feat_arr, val_type=0)
kbest, qs = recommender.compute_kbest_type(corr_loc, count_matrix, feat_arr, kbest=5)

