
"""
Region Retrievers
-----------------


"""

import numpy as np
from scipy.spatial.distance import cdist

from retrievers import Retriever


###############################################################################
############################## Region Retrievers ##############################
###############################################################################
class RegionRetriever(Retriever):
    """Retriever class for region-based retrievers.

    TODO:
    -----
    Non-unique discretization.

    """
    typeret = 'region'
    default_ret_val = {}
    locs_r = None

    def __init__(self, locs, discretizors, distance_reg, pars_ret=None,
                 autolocs=None, info_ret=None, flag_auto=True,
                 ifdistance=False, info_f=None, precomputed=True,
                 regiontype_input=True, ifcdist=True, regiontype_output=True):
        "Creation a point retriever class method."
        # Retrieve information
        pars_ret = self.format_pars_ret(pars_ret, precomputed)
        self.retriever = define_Regretriever(locs, discretizors, distance_reg,
                                             **pars_ret)
        self.info_ret = self.default_ret_val if info_ret is None else info_ret
        self.info_f = info_f
        self.ifdistance = ifdistance
        # Location information
        self.autolocs = True if autolocs is None else False
        self.locs = None if autolocs is None else autolocs
        self.locs_r = self.discretize(self.locs) if precomputed else None
        # Input-Output information
        self.flag_auto = flag_auto
        self.regiontype_input = True
        self.regiontype_output = True
        self.ifcdist = True
        # Extra info
        self.precomputed = precomputed

    def discretize(self, i_locs, unique=True):
        """This function transform the input i_locs to discs_i. This discretize
        function at this level has to deal with different inputs: (i_reg,
        i_locs, locs_i) and different possibilities: (precomputed, autolocs)

        Parameters
        ----------
        i_locs: int, numpy.ndarray, list or int or list of numpy.ndarray
            the location expressed as regions, id_points or the coordinates of
            the points.

        Returns
        -------
        discs_i: int, list of ints
            the region ids of the locations specified.

        """
        if i_locs is None:
            return None
        if self.regiontype_input:
            discs_i = i_locs
        else:
            if not self.check_coord(i_locs):
                if self.precomputed:
                    if self.autolocs:
                        discs_i = self.retriever.locs_r[i_locs]
                    else:
                        discs_i = self.locs_r[i_locs]
                else:
                    if self.autolocs:
                        locs_i = self.retriever.data[i_locs, :]
                    else:
                        locs_i = self.locs[i_locs, :]
                    discs_i = self.retriever.discretize(locs_i)
            else:
                discs_i = self.retriever.discretize(i_locs)
        return discs_i

    def retrieve_neighs_spec(self, loc_i, info_i, ifdistance=False):
        """Function of region retrieving.

        Parameters
        ----------
        loc_i: int or numpy.ndarray
            the location or the point id.
        info_i: optional
            information of retrieving points.
        ifdistance: boolean
            compute distance of the neighbours.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood regions.
        dists: numpy.ndarray
            the distances between regions.

        """
        ## Discretize location i
        regs_i = self.discretize(loc_i)
        ## Retrieve neighbourhood regions
        neighs, dists = self.retrieve_neigh_r(regs_i, info_i, ifdistance)
        return neighs, dists

    def retrieve_neigh_r(self, regs_i, info_i={}, ifdistance=False):
        """Retrieve regions neighbourhood.
        """
        info_i = self.retriever.format_info_i_reg(info_i)
        neighs_r, dists_r = self.retriever.retrieve_neigh_reg(regs_i, **info_i)
        dists_r = dists_r if ifdistance else None
        return neighs_r, dists_r

    def format_output(self, i_loc, neighs, dists):
        """Function which acts as a formatter of the output required by the
        design of the retriever object.

        Parameters
        ----------
        i_loc: int or numpy.ndarray
            the location or the point id.
        neighs: numpy.ndarray
            the ids of the neighbourhood regions.
        dists: numpy.ndarray
            the distances between regions.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood points or regions.
        dists: numpy.ndarray
            the distances between points or regions.

        """
        ## Format output shape
        if not self.regiontype_output:
            n = self.retriever.data.shape[0]
            logi = np.zeros(n).astype(bool)
            dists_i = np.zeros(n)
            for j in range(len(neighs)):
                logi_j = self.locs_r == neighs[j]
                logi = np.logical_or(logi, logi_j)
                dists_i[logi_j] = dists[j]
            neighs = np.where(logi)[0]
            dists = dists_i[logi]
        # Exclude auto point
        neighs, dists = self.exclude_auto(i_loc, neighs, dists)
        # Compute point-point distance
        point_i = self.get_loc_i(i_loc)
        if self.ifcdist:
            dists = cdist(point_i, self.retriever.data[neighs, :])
        return neighs, dists

    def format_pars_ret(self, pars_ret, precomputed):
        pars_ret = {'precomputed': precomputed}
        return pars_ret


def define_Regretriever(locs, discretizor, distance_reg, precomputed=True):
    regretriever = RegionNeighbourhood(locs, discretizor, distance_reg,
                                       precomputed)
    return regretriever
