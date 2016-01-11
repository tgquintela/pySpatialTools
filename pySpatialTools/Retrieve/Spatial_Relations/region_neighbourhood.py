
"""
Neighbourhood definition module
-------------------------------
This module contains the class which performs the neighbourhood retrieval from
spatial data regarding possible aggregation.
Deals with regions aggregation interface and retrieve region neighbourhood.


TODO
----
Join into the Region Retriever if it is possible.

"""

import numpy as np


class RegionNeighbourhood:
    """Retriever of regions given a discretization and a region.
    """

    data = None
    locs_r = []
    discretizors = []
    distance_reg = []
    regs = []
    reg_distances = []

    def __init__(self, locs, discretizors, distance_reg, precomputed=True):
        """
        Parameters
        ----------
        locs: numpy.ndarray
            the location coordinates.
        discretizors: list of spatialdisc objects or list of numpy.ndarray
            a list of discretization measures.
        distance_reg: list of regiondistances objects
            the information to compute the distance between regions.
        """
        m = len(discretizors)
        ## Location
        self.data = locs
        ## Discretization by regions
        if type(discretizors[0]) != np.ndarray:
            self.locs_r = [discretizors.discretize(locs, m) for i in range(m)]
            self.discretizors = None
        else:
            self.locs_r = discretizors
        self.regs = [np.unique(self.locs_r[m]) for i in range(m)]
        self.precomputed = precomputed
        ## Region distances
        self.distance_reg = distance_reg

    def discretize(self, i_locs, i_disc=0):
        """Discretization of location to retrieve region of which the point
        belongs.

        Parameters
        ----------
        i_locs: int, numpy.ndarray
            the information of the points to discretize.
        i_disc: int
            the discretization we want to apply.

        Returns
        -------
        discs_i: int or numpy.ndarray
            the regions of each point discretized.

        """
        if self.locs_r is None:
            discs_i = self.discretizors[i_disc].discretize(i_locs)
        else:
            discs_i = self.locs_r[i_disc][i_locs]
        return discs_i


class LimDistanceRegNeigh(RegionNeighbourhood):
    """Region Neighbourhood based on the limit distance bound.
    """

    def retrieve_neigh_reg(self, regs_i, i_disc=0, lim_distance=None,
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
        if regs_i not in self.regs:
            neighs = np.array([])
            dists = np.array([])
        else:
            neighs, dists = self.distance_reg.get_relations(regs_i)
            if lim_distance is None:
                logi = np.ones(len(list(dists))).astype(bool)
            else:
                if maxif:
                    logi = dists < lim_distance
                else:
                    logi = dists > lim_distance
            neighs = neighs[logi]
            dists = dists[logi]
        return neighs, dists

    def format_info_i_reg(self, info_i):
        pass


class SameRegNeigh(RegionNeighbourhood):
    """Region Neighbourhood based on the same region.
    """
    def retrieve_neigh_reg(self, regs_i, i_disc=0):
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
        neighs, dists = np.array([regs_i]), np.array([0])
        return neighs, dists

    def format_info_i_reg(self, info_i):
        pass


class OrderRegNeigh(RegionNeighbourhood):
    """Region Neighbourhood based on the order it is away from region
    direct neighbours.

    """
    exactorlimit = False

    def retrieve_neigh_reg(self, reg_i, i_disc=0, order=0, exactorlimit=False):
        """Retrieve the regions which are defined by the parameters of the
        inputs and the nature of this object method.

        Parameters
        ----------
        reg_i: int
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
        ## If not in regions
        if reg_i not in self.regs:
            neighs = np.array([])
            dists = np.array([])
        ## 0. Needed variables
        neighs, to_reg = [], [reg_i]
        ## Crawling in net variables
        to_reg, to_dists = [reg_i], [self.distance_reg.inv_null_value]
        reg_explored = []

        ## 1. Crawling in network
        for o in range(order+1):
            neighs_o, dists_o = [], []
            for i in range(len(to_reg)):
                neighs_o, dists_o = self.distance_reg.get_relations(to_reg[i])
                if self.distance_reg.distanceorweighs:
                    dists_o += to_dists[i]
                else:
                    dists_o += 1
            ## Add to globals
            n_o = len(neighs_o)
            idx_excl = [i for i in range(n_o) if to_reg[i] not in reg_explored]
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

        ## Transform to id
        neighs = [self.regs[neigh] for neigh in neighs]

        return neighs, dists

    def format_info_i_reg(self, info_i):
        pass
