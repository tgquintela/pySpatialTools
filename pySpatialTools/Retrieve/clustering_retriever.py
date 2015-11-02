
"""
Retrieve cluster
----------------
Module for clustering retrieving.


TODO:
----
- cluster_regions function

"""

import numpy as np

from pySpatialTools.Spatial_Relations.region_spatial_relations import \
    regions_relation_points


def clustering_regions(points, discretizor, retrievers, info_rets, clustering):
    """Pipeline function for spatial clustering points.

    Returns
    -------
    points_cl: numpy.ndarray
        array with the associated cluster for each point.

    """

    ## Needed variables
    m_ret = len(retrievers)

    ## Discretization
    discretizor.discretize(points)
    points_reg = discretizor.to_regions()  # TODO: function in discretize
    n_reg = np.unique(points_reg).shape[0]

    ## Creation of relation matrices
    coincidences = np.zeros((n_reg, n_reg, m_ret))
    for i in range(m_ret):
        coincidences[:, :, i] = regions_relation_points(points, points_reg,
                                                        retrievers[i],
                                                        info_rets[i])

    ## Clustering
    clusters = cluster_regions(coincidences, clustering)
    points_cl = np.zeros(points.shape[0])
    for i in range(clusters.shape[0]):
        points_cl[clusters == i] = i

    return points_cl


def measure_internal_prop(points, internal, retriever, info_i):
    """Measure the internal proportion of neighbours each point has.
    """

    ## Assignation
    proportion = np.zeros(points.shape[0])
    for i in xrange(points.shape[0]):
        neighs = retriever.retrieve_neighs(points[i, :], info_i)
        proportion[i] = np.mean(internal[neighs])

    return proportion


def cluster_regions(coincidences, clustering, params_cl):
    """Function which uses network clustering for cluster regions.

    TODO
    """
    ## Collapsing coincidences

    ## Clustering
    clusters = clustering.cluster(coincidences, **params_cl)

    return clusters
