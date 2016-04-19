
"""
Metric discretizer
------------------
Metric discretizer are all the discretizers based in a metric space.

"""

import numpy as np
from spatialdiscretizer import SpatialDiscretizor


class MetricDiscretizor(SpatialDiscretizor):
    """Metric discretizor groups all the applications, parameters and functions
    common of all discretizers based on metric spaces.
    """
    metric = True
    format_ = 'implicit'

    def map_locs2regionlocs(self, locs):
        "Map locations to regionlocs."
        regionid = self.discretize(locs)
        regionlocs = self._map_regionid2regionlocs(regionid)
        return regionlocs

    def map2agglocs_pre(self, locs):
        n_locs = locs.shape[0]
        agglocs = np.zeros(locs.shape).astype(float)
        regions = self.discretize(locs)
        # Average between all the locs circles
        for i in xrange(n_locs):
            agglocs[i, :] = np.mean(self.regionlocs[regions[i], :], axis=0)
        return agglocs

    def map2agglocs(self, locs):
        n_locs = locs.shape[0]
        agglocs = np.zeros(locs.shape).astype(float)
        regions = self.discretize(locs)
        # Average between all the locs circles
        for i in xrange(n_locs):
            agglocs[i, :] = np.mean(locs[regions == regions[i]], axis=0)
        return agglocs

#        if geom is True:
#            regionlocs = self._map_regionid2regionlocs(regions)
#        else:
#            regionslocs = np.zeros((regions.shape[0], elements.shape[1]))
#            for i in xrange(regions.shape[0]):
#                regionslocs[i, :] = elements[regions[i] == regions, :].mean(0)
