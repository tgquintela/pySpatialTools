
"""
Example EM
----------
Example using pySpatialTools to perform Expectation-Maximization algorithm by
using k-means algorithm.

TODO
----
* Cross cluster

"""

import numpy as np
from scipy.interpolate import Rbf
from pySpatialTools.base import BaseRelativePositioner, BaseDescriptorModel
from pySpatialTools.Retrieve import KRetriever, CircRetriever
from pySpatialTools.FeatureManagement.features_objects import ImplicitFeatures
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.spatial_descriptormodels import\
    SpatialDescriptorModel


class Weighted_RelativePositioner(BaseRelativePositioner):
    """funct = tuple with n-dim rbf transformations."""
    def compute(self, loc_i, loc_neighs):
        relative_xy = loc_neighs-loc_i
        n_dim = relative_xy.shape[1]
        dist = np.zeros(len(loc_neighs))
        for dim in range(n_dim):
            dist += np.divide(relative_xy[:, dim]**2,
                              self.funct[dim](loc_i[:, dim]))
        dist = dist.reshape((len(dist), 1))
        return dist


def _output_map_mindist_filter(retriever_o, i_locs, neighs_info):
    ## TODO: Change that (getloc)
    neighs, dists = [], []
    for i in range(len(i_locs)):
        idxs_i = np.argmin(np.sum(neighs_info[1]**2, axis=1).ravel())
        neighs.append([neighs_info[0][i][idxs_i]])
        dists.append([retriever_o.get_loc_i(neighs_info[0][i][idxs_i])])
    if retriever_o.constant_neighs:
        neighs = np.array(neighs)
        dists = np.array(dists)
    return (neighs, dists)


def _output_map_minrelpos_filter(retriever_o, i_locs, neighs_info):
    ## TODO: Change that (getloc)
    neighs, dists = [], []
    for i in range(len(i_locs)):
        idxs_i = np.argmin(neighs_info[1].ravel())
        neighs.append([neighs_info[0][i][idxs_i]])
        dists.append([retriever_o.get_loc_i(neighs_info[0][i][idxs_i])])
    if retriever_o.constant_neighs:
        neighs = np.array(neighs)
        dists = np.array(dists)
    return (neighs, dists)


class AvgPosition(BaseDescriptorModel):
    """"""
    def compute(self, pointfeats, point_pos):
        print pointfeats, point_pos
        descriptors = np.zeros((len(self.globals_), 2))
        pointfeats = np.concatenate(list(pointfeats))
        if all([e is None for e in point_pos]):
            return np.zeros(len(self.globals_)*3)
        point_pos = np.concatenate(list(point_pos))
        suma = np.zeros(len(self.globals_))
        for i in range(len(descriptors)):
            logi = pointfeats == self.globals_[i]
            descriptors[i] = np.sum(point_pos[logi], axis=0)
            suma[i] += np.sum(logi)

        descriptors = np.concatenate([descriptors.ravel(), suma])
        return descriptors

    def set_global_info(self, features):
        self.globals_ = np.unique(features.ravel())

    def _f_default_names(self, features):
        names = [str(i) for i in np.arange(len(np.unique(features.ravel()))*3)]
        return names


class AvgJoinerPosition(AvgPosition):
    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        "Relative descriptors."
        return desc_neigh


if __name__ == '__main__':
    #### Case of equal axis variance
    ## Initialization of the points
    # Parameters
    nclusters = 3
    # Random clustered points
    x = np.concatenate([np.random.normal(5, size=500),
                        np.random.normal(-4, size=500),
                        np.random.normal(-3, 0.2, size=1000)])
    y = np.concatenate([np.random.normal(4, size=500),
                        np.random.normal(-3, size=500),
                        np.random.normal(0, 0.2, size=1000)])
    # Random initialization
    x0 = (np.random.random(nclusters)-0.5)*10
    y0 = (np.random.random(nclusters)-0.5)*8
    # Parameters algorithm
    n_steps = 100
    # Parameters
    points = np.vstack([x, y]).T
    # Initialization
    centroids = (np.random.random((nclusters, 2))-.5)*10
    selectors = ((0, 0), (0, 1), (0, 0))
    avgdesc = AvgPosition()
    avgdesc.set_global_info(np.arange(nclusters))
    feats = ImplicitFeatures(np.arange(nclusters), descriptormodel=avgdesc)
    ## Performing algorithm
    for i in range(n_steps):
        ## Expectation (assignation of the points)
        ret0 = KRetriever(locs=centroids, autolocs=points, info_ret=nclusters,
                          ifdistance=True,
                          output_map=_output_map_mindist_filter)
        feats_ret = FeaturesManager([points, feats],
                                    maps_vals_i=np.zeros(len(points)),
                                    selectors=selectors,
                                    descriptormodels=AvgJoinerPosition())
        spdesc = SpatialDescriptorModel(ret0, feats_ret)
        measure = spdesc.compute()

    #### Case of different axis variance
    ## Initialization of the points
    # Parameters
    nclusters = 3
    # Random clustered points
    x = np.concatenate([np.random.normal(5, 0.2, size=500),
                        np.random.normal(-4, 2.0, size=500),
                        np.random.normal(-3, 2.4, size=1000)])
    y = np.concatenate([np.random.normal(4, 5, size=500),
                        np.random.normal(-3, 0.4, size=500),
                        np.random.normal(0, 0.2, size=1000)])
    # Random initialization
    x0 = (np.random.random(nclusters)-0.5)*10
    y0 = (np.random.random(nclusters)-0.5)*8
    # Parameters algorithm
    n_steps = 100
    # Interpolate metric weights
    sigma_x, sigma_y = np.array([0.2, 2.0, 2.4]), np.array([5., 0.4, 0.2])
    center_x, center_y = np.array([5., -4., -3.]), np.array([4., -3., 0.])
    rbf_x = Rbf(center_x, center_y, sigma_x, epsilon=2)
    rbf_y = Rbf(center_x, center_y, sigma_y, epsilon=2)

    ## Performing algorithm
    # Parameters
    relative_pos = Weighted_RelativePositioner((rbf_x, rbf_y))
    points = np.vstack([x, y]).T
    # Initialization
    centroids = (np.random.random((nclusters, 2))-.5)*10
    feats = ImplicitFeatures(np.arange(nclusters),
                             descriptormodel=AvgPosition())
    for i in range(n_steps):
        ## Expectation (assignation of the points)
        ret0 = KRetriever(locs=centroids, autolocs=points, info_ret=nclusters,
                          ifdistance=True, relative_pos=relative_pos,
                          output_map=_output_map_minrelpos_filter)
        feats_ret = FeaturesManager(feats)
        spdesc = SpatialDescriptorModel(ret0, feats_ret)
