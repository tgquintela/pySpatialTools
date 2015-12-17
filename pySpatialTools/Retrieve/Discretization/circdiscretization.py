
"""
Circular discretization
-----------------------
Module which groups classes and functions related with circular based
discretization of space.

"""

import numpy as np
from spatialdiscretizer import SpatialDiscretizor
#from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from pythonUtils.parallel_tools import distribute_tasks


############################### Circular based ################################
###############################################################################
class CircularSpatialDisc(SpatialDiscretizor):
    """Circular spatial discretization. The regions are circles with different
    sizes. One point could belong to zero, one or more than one region.
    """
    multiple_regions = False

    ## TODO: map loc_grid to a id region: map_gridloc2regionid
    def __init__(self, centerlocs, radios, multiple_regions=False):
        "Main information to built the regions."
        if type(radios) in [float, int]:
            radios = np.ones(centerlocs.shape[0])*radios
        self.borders = radios
        self.regionlocs = centerlocs

    ################################ Functions ###############################
    ##########################################################################
    def map_loc2regionid(self, locs):
        """Discretize locs returning their region_id.

        Parameters
        ----------
        locs: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regionid: numpy.ndarray
            the region_id of each locs for this discretization.

        """
        regionid = map_circloc2regionid(locs, self.regionlocs, self.borders,
                                        self.multiple_regions)
        return regionid

    def map_regionid2regionlocs(self, regions):
        """Function which maps the regions ID to their most representative
        location.
        """
        regionlocs = np.zeros((regions.shape[0], self.regionslocs.shape[1]))
        for i in xrange(regions.shape):
            idx = np.where(self.regions_id == regions[i])[0][0]
            regionlocs[i, :] = self.regionlocs[idx, :]
        return regionlocs

    def map_locs2regionlocs(self, locs):
        "Map locations to regionlocs."
        regionid = self.map_loc2regionid(locs)
        regionlocs = self.map_regionid2regionlocs(regionid)
        return regionlocs

    def map2aggloc_spec(self, locs):
        n_locs = locs.shape[0]
        agglocs = np.zeros(locs.shape).astype(float)
        regions = self.discretize(locs)
        # Average between all the locs circles
        for i in xrange(n_locs):
            agglocs[i, :] = np.mean(self.regionlocs[regions[i], :], axis=0)
        return agglocs

    def compute_limits(self, region_id=None):
        if region_id is None:
            limits = compute_limits_circ(self.regionslocs, self.borders,
                                         self.regions_id)
            self.limits = limits
        else:
            limits = compute_limits_circ(self.regionslocs, self.borders,
                                         self.regions_id, region_id)
            return limits

    def compute_contiguity_geom(self, region_id=None):
        pass


def compute_limits_circ(regionslocs, radis, regions_id, regionid=None):
    limits = np.zeros((2, 2))
    if regionid is None:
        idx1, idx2 = np.argmin(regionslocs[:, 0]), np.argmax(regionslocs[:, 0])
        limits[0, 0] = regionslocs[idx1, 0] - radis[idx1]
        limits[0, 1] = regionslocs[idx2, 0] + radis[idx2]
        idx1, idx2 = np.argmin(regionslocs[:, 1]), np.argmax(regionslocs[:, 1])
        limits[1, 0] = regionslocs[idx1, 1] - radis[idx1]
        limits[1, 1] = regionslocs[idx2, 1] + radis[idx2]
    else:
        idx = np.where(regions_id == regionid)[0][0]
        limits[0, 0] = regionslocs[idx, 0] - radis[idx]
        limits[0, 1] = regionslocs[idx, 0] + radis[idx]
        limits[1, 0] = regionslocs[idx, 1] - radis[idx]
        limits[1, 1] = regionslocs[idx, 1] + radis[idx]

    return limits


def map_circloc2regionid(locs, centerlocs, radis, multiple=False):
    "Map each point to the correspondent circular region."
    ## 0. If multiple regions assignation it is allowed
    if multiple:
        regions_id = map_circloc2regionid_multiple(locs, centerlocs, radis)
        return regions_id
    ## 1. Computation of indices
    # Fraction of work
    idxs_dis = distribute_tasks(locs.shape[0], 50000)
    regions_id = np.zeros(locs.shape[0])
    for k in range(len(idxs_dis)):
        # Computation of proportion in the radis
        dists = cdist(locs[idxs_dis[k][0]:idxs_dis[k][1]], centerlocs)
        prop = (dists - radis) / radis
        idxs = np.argmin(prop, axis=1)
        boolean = prop[range(idxs.shape[0]), idxs] > 0
        # Assignation of the indices
        regions_id[idxs_dis[k][0]:idxs_dis[k][1]] = idxs
        regions_id[idxs_dis[k][0]:idxs_dis[k][1]][boolean] = -1

    regions_id = regions_id.astype(int)

    return regions_id


def map_circloc2regionid_multiple(locs, centerlocs, radis):
    "Map the each point to the correspondent circular region."
    idxs_dis = distribute_tasks(locs.shape[0], 50000)
    regions_id = [[] for i in range(locs.shape[0])]
    for k in range(len(idxs_dis)):
        logi = cdist(locs[idxs_dis[k][0]:idxs_dis[k][1]], centerlocs) < radis
        aux = np.where(logi)
        for j in range(len(aux[0])):
            regions_id[aux[0][j]].append(aux[1][j])
    return regions_id
