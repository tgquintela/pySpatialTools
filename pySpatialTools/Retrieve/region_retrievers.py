
"""
Region Retrievers
-----------------
Region retriever module in which is stored all the region retrivers. The region
retrievers have the particularity to retrieve use regions to define
neighbourhoods and retrieve points or regions.
The retrieve could be auto-retrieve (retrieve the neighbours of some elements
from the same pool of elements) or cross-retrieve (retrieve elements from a
different pool of elements).


Compulsary requisits
--------------------
- discretize (function)
- retrieve_neighs_spec (function)
- format_output (function)
- retriever.data (parameters)

TODO
----
- discretize function
- relative_pos function

"""

import numpy as np
#from scipy.spatial.distance import cdist

from retrievers import Retriever


###############################################################################
########################### Region-Point Retrievers ###########################
###############################################################################
class RegionPointRetriever(Retriever):
    """Retriever class which use regions to compute the neighbourhood.
    """

    typeret = 'region_point'
    default_ret_val = {}

    locs = None
    locs_r = None
    autolocs = True

    def __init__(self, regionretriever, locs, discretizor, autolocs=None,
                 pars_ret=None, info_ret=None, info_f=None, ifdistance=False,
                 flag_auto=True, relative_pos=None, precomputed=True):
        "Creation a point retriever class method."
        # Retriever information
        #pars_ret = self.format_pars_ret(pars_ret, precomputed)
        #self.retriever = define_RegPointretriever(regionretriever, **pars_ret)
        self.retriever = regionretriever
        self.info_ret = self.default_ret_val if info_ret is None else info_ret
        if type(info_ret).__name__ == 'function':
            self.info_f = info_ret
        else:
            self.info_f = info_f
        self.info_ret = self.default_ret_val if info_ret is None else info_ret
        self.ifdistance = ifdistance
        # Location information
        self.discretizor = discretizor
        self.retriever.data = locs
        self.retriever.data_r = self.discretizor.discretize(locs)
        self.locs = None if autolocs is None else autolocs
        self.autolocs = True if self.locs is None else False
        # Filter information
        self.flag_auto = flag_auto
        # Relative position object creator
        self.relative_pos = relative_pos

    def discretize(self, i_locs):
        "Discretization of locations."
        if type(i_locs) != int:
            if self.locs_r is None:
                i_locs = self.discretizor.discretize(i_locs)
            else:
                i_locs = self.locs_r[i_locs]
        return i_locs

    def retrieve_neighs_spec(self, point_i, info_i, ifdistance=False,
                             info_r={}, ifdistance_r=True):
        "Retrieve points neighbourhood."
        ## 0. Region neighbourhood retrieve
        i_disc = self.discretize(point_i)
        i_disc = np.array([i_disc])
        info_i = self.get_info_i(point_i, info_i)
        # Retrieve region neighbourhood
        neighs_r, dists_r = self.retriever.retrieve_neighs_spec(i_disc,
                                                                info_i,
                                                                ifdistance_r)
        ## 1. Point neighbourhood retrieve from regions
        neighs_i, neighs_ir, dists_ir = self.get_points_r(neighs_r, dists_r)
        # Relative positions computation
        if ifdistance:
            dists_i = self.compute_relative_i(point_i, neighs_i, neighs_ir,
                                              dists_ir)
        else:
            dists_i = None
        return neighs_i, dists_i

    def format_output(self, i_locs, neighs, dists):
        "Format output."
        neighs, dists = self.exclude_auto(i_locs, neighs, dists)
        return neighs, dists

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def get_points_r(self, neighs_r, dists_r):
        """Get spatial context information for points from regions. The task of
        this function is to replicate the distances and region neighbours
        obtained by the region retriever to the element retriever format (each
        element neigh his region and the relative position of its region).
        """
        if self.retriever.data_r is not None:
            neighs_i, neighs_ir, dists_ir = [], [], []
            for i in range(len(list(neighs_r))):
                nei = neighs_r[i]
                neighs_ii = np.where(self.retriever.data_r == nei)[0]
                neighs_i.append(neighs_ii)
                neighs_ir.append(nei*np.ones(neighs_ii.shape[0]).astype(int))
                dists_ir.append(dists_r[i]*np.ones(neighs_ii.shape[0]))
        else:
            locs_r = self.discretize(self.retriever.data)
            neighs_i, neighs_ir, dists_ir = [], [], []
            for i in range(len(list(neighs_r))):
                neighs_ii = np.where(locs_r == nei)[0]
                neighs_i.append(neighs_ii)
                neighs_ir.append(nei*np.ones(neighs_ii.shape[0]).astype(int))
                dists_ir.append(dists_r[i]*np.ones(neighs_ii.shape[0]))
        ## Correct output
        if len(list(neighs_i)) != 0:
            neighs_i, neighs_ir = np.hstack(neighs_i), np.hstack(neighs_ir)
            dists_ir = np.hstack(dists_ir)
        else:
            neighs_i, neighs_ir = np.array([]), np.array([])
            dists_ir = np.array([])
        return neighs_i, neighs_ir, dists_ir

    def compute_relative_i(self, i_loc, neighs_i, neighs_ir, dists_ir):
        "Compute relative information for point i."
        i_disc = self.discretize(i_loc)
        if self.relative_pos is not None:
            if type(self.relative_pos).__name__ == 'function':
                dists_i = self.relative_pos(i_loc, i_disc, neighs_i, neighs_ir,
                                            dists_ir)
            else:
                dists_i = self.relative_pos.compute_reg(i_loc, i_disc,
                                                        neighs_i, neighs_ir,
                                                        dists_ir)
        else:
            dists_i = dists_ir
        return dists_i

#
#def define_RegPointretriever(regionretriever, **pars_ret):
#    regionretriever.data = 0
#    pass


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

    def __init__(self, regionmetrics, pars_ret=None, info_ret=None,
                 flag_auto=False, ifdistance=True, info_f=None):
        "Creation a point retriever class method."
        # Retrieve information
        pars_ret = self.format_pars_ret(pars_ret)
        #self.retriever = define_Regretriever(regionmetrics, **pars_ret)
        self.retriever = regionmetrics
        self.info_ret = self.default_ret_val if info_ret is None else info_ret
        self.info_f = info_f
        self.ifdistance = ifdistance
        # Location information
        self.flag_auto = flag_auto

    def retrieve_neighs_spec(self, regs_i, info_i={}, ifdistance=False):
        """Retrieve regions neighbourhood.
        """
        info_i = self.format_info_i_reg(info_i, regs_i)
        neighs_r, dists_r = self.retrieve_neighs_reg(regs_i, **info_i)
        neighs_r = neighs_r.ravel()
        dists_r = dists_r if ifdistance else None
        return neighs_r, dists_r

    def format_pars_ret(self, pars_ret, precomputed=True):
        pars_ret = {'precomputed': precomputed}
        return pars_ret

    def format_output(self, i_locs, neighs, dists):
        "Format output."
        neighs, dists = self.exclude_auto(i_locs, neighs, dists)
        return neighs, dists

    def format_info_i_reg(self, info_i, i=-1):
        if bool(info_i):
            pass
        else:
            if type(i) == np.ndarray:
                i = np.where(self.retriever.data == i)[0]
                i = i[0] if i.any() else -1
            if self.info_ret is not None:
                if type(self.info_ret) == list:
                    if i != -1:
                        info_i = self.info_ret[i]
                    else:
                        info_i = {}
                elif type(self.info_ret) == dict:
                    info_i = self.info_ret
        return info_i

#
#def define_Regretriever(locs, discretizor, distance_reg, precomputed=True):
#    "TODEPRECATE"
#    regretriever = RegionNeighbourhood(locs, discretizor, distance_reg,
#                                       precomputed)
#    return regretriever


class LimDistanceRegNeigh(RegionRetriever):
    """Region Neighbourhood based on the limit distance bound.
    """

    def retrieve_neighs_reg(self, regs_i, i_disc=0, lim_distance=None,
                            maxif=True, ifdistance=True):
        """Retrieve the regions which are defined by the parameters of the
        inputs and the nature of this object method.

        Parameters
        ----------
        regs_i: int or numpy.ndarray
            the region we want to get its neighsbour regions.
        i_disc: int
            the discretization we want to apply.
        lim_distance: float
            the bound distance to define neighbourhood.
        maxif: boolean
            if True the bound is the maximum accepted, if False it is the
            minimum.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood regions.
        dists: numpy.ndarray
            the distances between regions.

        """
        neighs, dists = self.retriever.retrieve_neighs(regs_i)
        if neighs.any():
            if lim_distance is None:
                logi = np.ones(len(list(dists))).astype(bool)
            else:
                if maxif:
                    logi = dists < lim_distance
                else:
                    logi = dists > lim_distance
            neighs = neighs[logi]
            dists = dists[logi]
        neighs = neighs.ravel()
        return neighs, dists


class SameRegNeigh(RegionRetriever):
    """Region Neighbourhood based on the same region.
    """
    def retrieve_neighs_reg(self, regs_i, i_disc=0):
        """Retrieve the regions which are defined by the parameters of the
        inputs and the nature of this object method.

        Parameters
        ----------
        regs_i: int or numpy.ndarray
            the region we want to get its neighsbour regions.
        i_disc: int
            the discretization we want to apply.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood regions.
        dists: numpy.ndarray
            the distances between regions.

        """
        if type(regs_i) == np.ndarray:
            neighs, dists = np.array(regs_i), np.array([0])
        else:
            neighs = np.array(self.retriever.data[regs_i])
            dists = np.array([0]*neighs.shape[0])
        neighs = neighs.ravel()
        return neighs, dists


class OrderRegNeigh(RegionRetriever):
    """Region Neighbourhood based on the order it is away from region
    direct neighbours.

    """
    exactorlimit = False

    def retrieve_neighs_reg(self, reg_i, i_disc=0, order=0,
                            exactorlimit=False):
        """Retrieve the regions which are defined by the parameters of the
        inputs and the nature of this object method.

        Parameters
        ----------
        reg_i: int or np.ndarray
            the region we want to get its neighsbour regions.
        i_disc: int
            the discretization we want to apply.
        order: int
            the order we want to retrieve the object.
        exactorlimit: boolean
            if True we retrieve the neighs in this exact order, if False we
            retrieve the neighs up to this order (included).


        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood regions.
        dists: numpy.ndarray
            the distances between regions.

        """
        reg_i = self.retriever.data[reg_i] if type(reg_i) == int else reg_i
        ## 0. Needed variables
        neighs, dists, to_reg = [], [], [reg_i]
        ## Crawling in net variables
        to_reg, to_dists = [reg_i], [self.retriever.inv_null_value]
        reg_explored = []

        ## 1. Crawling in network
        for o in range(order+1):
            neighs_o, dists_o = [], []
            for i in range(len(to_reg)):
                to_reg_i = np.array(to_reg[i])
                neighs_o, dists_o = self.retriever.retrieve_neighs(to_reg_i)
                if self.retriever.distanceorweighs:
                    dists_o += to_dists[i]
                else:
                    dists_o += 1
            ## Add to globals
            n_o = len(neighs_o)
            idx_excl = [i for i in range(n_o) if neighs_o[i]
                        not in reg_explored]
            to_reg = [neighs_o[i] for i in range(n_o) if i not in idx_excl]
            to_dists = [dists_o[i] for i in range(n_o) if i not in idx_excl]
            reg_explored += to_reg
            ## Add to results
            neighs.append(neighs_o)
            dists.append(dists_o)

        ## Exact or limit order formatting output
        if exactorlimit:
            neighs = neighs_o
            dists = dists_o

        neighs, dists = np.hstack(neighs).ravel(), np.hstack(dists)
        return neighs, dists
