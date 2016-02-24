
"""
Circular discretization
-----------------------
Module which groups classes and functions related with circular based
discretization of space.

"""

import numpy as np
from scipy.spatial.distance import cdist

from ..metricdiscretizor import MetricDiscretizor
from pySpatialTools.utils.util_external import distribute_tasks


class CircularSpatialDisc(MetricDiscretizor):
    """Circular spatial discretization. The regions are circles with different
    sizes. One point could belong to zero, one or more than one region.
    """
    n_dim = 2

    def __init__(self, centerlocs, radios, regions_id=None,
                 multiple_regions=False):
        """Main information to built the regions."""
        self._initialization()
        if type(radios) in [float, int]:
            radios = np.ones(centerlocs.shape[0])*radios
        self.borders = radios
        self.regionlocs = centerlocs
        self._format_regionsid(regions_id)
        self._compute_limits()

    def _format_regionsid(self, regions_id):
        if regions_id is None:
            self.regions_id = np.arange(len(self.regionlocs))
        else:
            assert(len(regions_id) == len(self.regionlocs))
            self.regions_id = regions_id

    def _compute_limits(self, region_id=None):
        """Compute bounding box limits of the selected region or the whole
        discretization."""
        if region_id is None:
            limits = compute_limits_circ(self.regionlocs, self.borders,
                                         self.regions_id)
            self.limits = limits
        else:
            limits = compute_limits_circ(self.regionlocs, self.borders,
                                         self.regions_id, region_id)
            return limits

    def _map_loc2regionid(self, locs):
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
                                        self.multiple)
        return regionid

    def _compute_contiguity_geom(self, region_id=None):
        """Compute contiguity geometry between regions following the
        instructions passed through the given parameters."""
        # TODO:
        raise Exception("Not implemented function yet.")

    def _map_regionid2regionlocs(self, regions):
        """Function which maps the regions ID to their most representative
        location.
        """
        if type(regions) == int:
            regions = np.array([regions])
        regionlocs = np.zeros((regions.shape[0], self.regionlocs.shape[1]))
        for i in xrange(len(regions)):
            ## Only get the first one
            idx = np.where(self.regions_id == regions[i])[0]
            if len(idx) == 0:
                raise Exception("Region not in the discretization.")
            regionlocs[i, :] = self.regionlocs[idx[0], :]
        return regionlocs


############################### Circular based ################################
###############################################################################
class CircularExcludingSpatialDisc(CircularSpatialDisc):
    """Circular spatial discretization. The regions are circles with different
    sizes. One point only could belong to one region or anyone (-1).
    """
    multiple = False


class CircularInclusiveSpatialDisc(CircularSpatialDisc):
    """Circular spatial discretization. The regions are circles with different
    sizes. One point could belong to zero, one or more than one region.
    """
    multiple = True


################################## Functions #################################
##############################################################################
def compute_limits_circ(regionslocs, radis, regions_id, regionid=None):
    """Function to compute the limits in a circular base discretization."""
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
