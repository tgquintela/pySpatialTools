
"""
Retrieve cluster
----------------
Module for clustering retrieving.


TODO:
----
- cluster_regions function



Pipeline
--------
1. Compute spatial relations
2. Discretize points
3. Assign points to clusters

"""

import numpy as np

from pySpatialTools.SpatialRelations.region_spatial_relations import \
    regions_relation_points


## discretization --> feats
## descriptormodel -->
## do it as sparse or network


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
    n_reg = len(np.unique(points_reg))

    ## Creation of relation matrices
    coincidences = np.zeros((n_reg, n_reg, m_ret))
    for i in range(m_ret):
        coincidences[:, :, i] = regions_relation_points(points, points_reg,
                                                        retrievers[i],
                                                        info_rets[i])

    ## Clustering
    clusters = cluster_regions(coincidences, clustering)
    points_cl = np.zeros(len(points))
    for i in range(len(clusters)):
        points_cl[clusters == i] = i

    return points_cl


def measure_internal_prop(internal, retriever, info_i=None):
    """Measure the internal proportion of neighbours each point has.
    """
    ## Only internal points
    indices = np.where(internal)[0]
    ## Assignation
    proportion = np.zeros(len(indices))
    for i in xrange(len(indices)):
        neighs = retriever.retrieve_neighs(indices[i], info_i)
        proportion[i] = np.mean(internal[neighs])
    return proportion


#def cluster_regions(coincidences, clustering, params_cl):
#    """Function which uses network clustering for cluster regions.
#
#    TODO
#    """
#    ## Collapsing coincidences
#
#    ## Clustering
#    clusters = clustering.cluster(coincidences, **params_cl)
#
#    return clusters

def points2clusters(points_regs, spatial_relations, clustering):
    "Transform points discretized to clusters."
    ## TODO
    clusters = clustering.cluster(spatial_relations.relations)
    ## Assign points clusters
    return points_cl
