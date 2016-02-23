
"""
sklearnlikediscretization
-------------------------

['n_dim', 'metric', 'format_']
    required_f = ['_compute_limits', '_compute_contiguity_geom',
                  '_map_loc2regionid', '_map_regionid2regionlocs']

"""

from ..metricdiscretizor import MetricDiscretizor


class SklearnDisc(MetricDiscretizor):
    """
    """

    def __init__(self, clf, limits):
        self._preformat(clf)
        self._initialization()
        self._format_clf(clf)
        self.limits = limits

    def _format_clf(self, clf):
        """Format clf to be used as discretizor."""
        if not "predict" in dir(clf):
            raise TypeError("Incorrect sklearn cluster method.")
        self.clf = clf

    def _map_regionid2regionlocs(self, region_id):
        pass

    def _map_loc2regionid(self, locations):
        """Discretize locations returning their region_id.

        Parameters
        ----------
        locations: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regions_id: numpy.ndarray
            the region_id of each location for this discretization.

        """
        return self.clf.predict(locations)

    def _compute_contiguity_geom(self, limits):
        "Compute which regions are contiguous and returns a graph."
        ## TODO:
        raise Exception("Not implemented function yet.")
        ## Obtain random points around all the r_points
        ## Compute the two nearest points with different region_id
        ## Remove repeated pairs
        return

    def _compute_limits(self):
        pass
