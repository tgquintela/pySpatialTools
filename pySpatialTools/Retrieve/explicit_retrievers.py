
"""
Explicit retrievers
------------------
The module which groups the explicit defined retrievers of generic elements.

"""

import numpy as np
from itertools import combinations
from retrievers import Retriever


###############################################################################
############################## Network Retrievers #############################
###############################################################################
class NetworkRetriever(Retriever):
    """Retriever class for precomputed network distances.
    """
    typeret = 'network'
    _default_ret_val = {}

    def __init__(self, main_mapper, info_ret=None, pars_ret=None,
                 flag_auto=True, ifdistance=True, info_f=None,
                 relative_pos=None, input_map=None, output_map=None):
        "Creation a element network retriever class method."
        # Reset globals
        self._initialization()
        ## Retrieve information
        self._define_retriever(main_mapper)
        ## Info_ret mangement
        self._format_retriever_info(info_ret, info_f)
        # Output information
        self._flag_auto = flag_auto
        self._ifdistance = ifdistance
        self.relative_pos = relative_pos
        # IO mappers
        self._format_maps(input_map, output_map)

    ############################## Main functions #############################
    ###########################################################################
    def _retrieve_neighs_spec(self, elem_i, info_i={}, ifdistance=False, kr=0):
        """Retrieve element neighbourhood information.
        """
        info_i = self._format_info_i_reg(info_i, elem_i)
        neighs, dists = self._retrieve_neighs_spec2(elem_i, kr=kr, **info_i)
        neighs = neighs.ravel()
        dists = dists if ifdistance else [[] for i in range(len(neighs))]
        return neighs, dists

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def _format_maps(self, input_map, output_map):
        if input_map is not None:
            self._input_map = input_map
        if output_map is not None:
            self._output_map = output_map

    def _define_retriever(self, main_mapper, pars_ret={}):
        "Define the main mapper as a special retriever."
        self.retriever.append(main_mapper)

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        check = True
        return check

    def _format_output(self, i_locs, neighs, dists, output, k=0):
        "Format output."
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, k)
        neighs, dists = np.array(neighs), np.array(dists)
        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
        dists = dists.reshape((len(dists), n_dim))
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs, dists

    def _format_info_i_reg(self, info_i, i=-1):
        "TODO: match index from relations."
        if bool(info_i):
            pass
        else:
            info_i = self._info_ret
#            if type(i) == np.ndarray:
#                i = np.where(self.retriever.data == i)[0]
#                i = i[0] if i.any() else -1
#            if self._info_ret is not None:
#                if type(self._info_ret) == list:
#                    if i != -1:
#                        info_i = self._info_ret[i]
#                    else:
#                        info_i = {}
#                elif type(self._info_ret) == dict:
#                    info_i = self._info_ret
        return info_i


class LimDistanceEleNeigh(NetworkRetriever):
    """Region Neighbourhood based on the limit distance bound.
    """
    _default_ret_val = {}

    def _retrieve_neighs_spec2(self, elem_i, lim_distance=None, maxif=True,
                               ifdistance=True, kr=0):
        """Retrieve the elements which are defined by the parameters of the
        inputs and the nature of this object method. This function retrieve
        neighbours defined by the map object defined in the parameter retriever
        and filter regarding the limit distance defined.

        Parameters
        ----------
        elem_i: int or numpy.ndarray
            the elements we want to get its neighbor element.
        lim_distance: float
            the bound distance to define neighbourhood.
        maxif: boolean
            if True the bound is the maximum accepted, if False it is the
            minimum.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood elements.
        dists: numpy.ndarray
            the distances between elements.

        """
        neighs, dists = self.retriever[kr][elem_i]
        if neighs.any():
            if lim_distance is None:
                logi = np.ones(len(dists)).astype(bool)
            else:
                if maxif:
                    logi = dists < lim_distance
                else:
                    logi = dists > lim_distance
            neighs = neighs[logi]
            dists = dists[logi]
        neighs = neighs.ravel()
        return neighs, dists


class SameEleNeigh(NetworkRetriever):
    """Network retriever which returns the same element as the mapper defined
    in the retriever maps.
    """
    _default_ret_val = {}

    def _retrieve_neighs_spec2(self, elem_i, kr=0):
        """Retrieve the elements which are defined by the parameters of the
        inputs and the nature of this object method.

        Parameters
        ----------
        elem_i: int or numpy.ndarray
            the element we want to get its neighsbour elements.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood elements.
        dists: numpy.ndarray
            the distances between elements.

        """
        neighs, dists = self.retriever[kr][elem_i]
        neighs = neighs.ravel()
        return neighs, dists


class OrderEleNeigh(NetworkRetriever):
    """Network retriever based on the order it is away from element
    direct neighbours in a network.
    """
    _default_ret_val = {}

    def _retrieve_neighs_spec2(self, elem_i, order=0, exactorlimit=False,
                               kr=0):
        """Retrieve the elements which are defined by the parameters of the
        inputs and the nature of this object method.

        Parameters
        ----------
        elem_i: int or np.ndarray
            the element we want to get its neighsbour elements.
        order: int
            the order we want to retrieve the object.
        exactorlimit: boolean
            if True we retrieve the neighs in this exact order, if False we
            retrieve the neighs up to this order (included).

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood elements.
        dists: numpy.ndarray
            the distances between elements.

        """
        ## 0. Needed variables
        neighs, dists = [], []
        # Crawling in net variables
        to_reg, to_dists = [elem_i], [self.retriever[kr]._inv_null_value]
        reg_explored = []

        ## 1. Crawling in network
        for o in range(order+1):
            neighs_o, dists_o = [], []
            for i in range(len(to_reg)):
                to_reg_i = np.array(to_reg[i])
                neighs_oi, dists_oi = self.retriever[kr][to_reg_i]
                if self.retriever[kr]._distanceorweighs:
                    dists_oi = dists_oi.astype(float)
                    dists_oi += to_dists[i]
                else:
                    dists_oi += 1
                # Append to neighs_o and dists_o
                neighs_o.append(neighs_oi)
                dists_o.append(dists_oi)
            ## Discard repeated neighs in neighs_o
            idx_excl = []
            for i, j in combinations(range(len(neighs_o)), 2):
                if neighs_o[i] == neighs_o[j]:
                    logi = dists_o[i] <= dists_o[j]
                    if logi:
                        idx_excl.append(j)
                    else:
                        idx_excl.append(i)
            n_o = len(neighs_o)
            dists_o = [dists_o[i] for i in range(n_o) if i not in idx_excl]
            neighs_o = [neighs_o[i] for i in range(n_o) if i not in idx_excl]

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
        neighs_, dists_ = [], []
        for i in range(len(neighs)):
            neighs_ += neighs[i]
            dists_ += dists[i]
        neighs, dists = neighs_, dists_
        neighs, dists = np.hstack(neighs).ravel(), np.hstack(dists)
        return neighs, dists
