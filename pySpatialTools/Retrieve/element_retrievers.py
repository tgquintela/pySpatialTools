
"""
Element_retrievers
------------------
The module which contains a retriever of generic elements.

"""

from retrievers import Retriever
import numpy as np
from sklearn.neighbors import KDTree
from itertools import combinations


###############################################################################
############################# Element Retrievers ##############################
###############################################################################
class ElementRetriever(Retriever):
    """Retreiver of elements given other elements and only considering the
    information of the non-retrivable elements.
    """

    typeret = 'element'

    ## Elements information
    data = None
    _autodata = False
    ## Retriever information
    retriever = None
    _info_ret = None
    _info_f = None
    ## External objects to apply
    relative_pos = None
    ## IO information
    _flag_auto = False
    _ifdistance = False
    _autoret = False
    _heterogenous_input = False
    _heterogenous_output = False
    ## IO methods
    _input_map = None
    _output_map = None

    def discretize(self, i_locs):
        """Format the index retrieving for the proper index of retrieving of
        the type of retrieving.
        """
        if self._input_map is not None:
            i_locs = self._input_map[i_locs]
        else:
            if self.check_coord(i_locs):
                if type(i_locs) == list:
                    i_locs = -1 * np.ones(len(i_locs))
                else:
                    i_locs = -1
        return i_locs

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def _format_output(self, i_locs, neighs, dists):
        "Format output."
        neighs, dists = self._exclude_auto(i_locs, neighs, dists)
        ## If not auto not do it
        return neighs, dists

    def _check_relative_position(self, relative_position, neighs):
        "Check if the relative position computed is correct."
        if not len(neighs) == len(relative_position):
            raise Exception("Not correct relative position computed.")


###############################################################################
############################ Space-Based Retrievers ###########################
###############################################################################
class SpaceRetriever(Retriever):
    """Retriever of elements considering its spacial information from a pool
    of elements retrievable.
    """

    typeret = 'space'

    def __init__(self, locs, info_ret=None, autolocs=None, pars_ret=None,
                 flag_auto=True, ifdistance=False, info_f=None,
                 relative_pos=None, input_map=None, output_map=None):
        "Creation a element space retriever class method."
        ## Retrieve information
        self._define_retriever(locs, pars_ret)
        ## Info_ret mangement
        self._format_retriever_info(info_ret, info_f)
        # Location information
        self._format_locs(locs, autolocs)
        # Output information
        self._flag_auto = flag_auto
        self._ifdistance = ifdistance
        self.relative_pos = relative_pos
        # IO mappers
        self._format_maps(input_map, output_map)

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def _format_maps(self, input_map, output_map):
        if input_map is not None:
            self._input_map = input_map
        if output_map is not None:
            self._output_map = output_map

    def _format_output(self, i_locs, neighs, dists, output):
        "Format output."
        neighs, dists = self._output_map[output](i_locs, (neighs, dists))
        neighs, dists = self._exclude_auto(i_locs, neighs, dists)
        return neighs, dists

    def _format_locs(self, locs, autolocs):
        self.data = None if autolocs is None else autolocs
        self._autodata = True if self.data is None else False

    def _format_neigh_info(self, neighs_info):
        "TODO: Extension for other type of data as shapepy objects."
        neighs, dists = np.array(neighs_info[0]), np.array(neighs_info[1])
        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
        dists = dists.reshape((len(dists), n_dim))
        return neighs, dists


################################ K Neighbours #################################
###############################################################################
class KRetriever(SpaceRetriever):
    "Class which contains a retriever of K neighbours."
    _default_ret_val = 1

    def _retrieve_neighs_spec(self, point_i, kneighs_i, ifdistance=False):
        "Function to retrieve neighs in the specific way we want."
        point_i = self._get_loc_i(point_i)
        res = self.retriever.query(point_i, int(kneighs_i), ifdistance)
        if ifdistance:
            res = res[1][0], res[0][0]
        else:
            res = res[0], [[] for i in range(len(res[0]))]
        ## Correct for another relative spatial measure
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever.data)[res[0], :]
            if type(self.relative_pos).__name__ == 'function':
                res = res[0], self.relative_pos(point_i, loc_neighs)
            else:
                res = res[0], self.relative_pos.compute(point_i, loc_neighs)
        return res

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        try:
            loc = self.retriever.data[0]
            self.retriever.query(loc, 2)
            check = True
        except:
            check = False
        return check

    def _define_retriever(self, locs, pars_ret=None):
        "Define a kdtree for retrieving neighbours."
        if pars_ret is not None:
            leafsize = int(pars_ret)
        else:
            leafsize = locs.shape[0]
            leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        self.retriever = KDTree(locs, leaf_size=leafsize)


################################ R disctance ##################################
###############################################################################
class CircRetriever(SpaceRetriever):
    "Circular retriever."
    _default_ret_val = 0.1

    def _retrieve_neighs_spec(self, point_i, radius_i, ifdistance=False):
        "Function to retrieve neighs in the specific way we want."
        point_i = self._get_loc_i(point_i)
        res = self.retriever.query_radius(point_i, radius_i, ifdistance)
        if ifdistance:
            res = res[0][0], res[1][0]
        else:
            res = res[0], [[] for i in range(len(res[0]))]
        ## Correct for another relative spatial measure
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever.data)[res[0], :]
            if type(self.relative_pos).__name__ == 'function':
                res = res[0], self.relative_pos(point_i, loc_neighs)
            else:
                res = res[0], self.relative_pos.compute(point_i, loc_neighs)
        return res

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        try:
            loc = self.retriever.data[0]
            self.retriever.query_radius(loc, 0.5)
            check = True
        except:
            check = False
        return check

    def _define_retriever(self, locs, pars_ret=None):
        "Define a kdtree for retrieving neighbours."
        if pars_ret is not None:
            leafsize = int(pars_ret)
        else:
            leafsize = locs.shape[0]
            leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        self.retriever = KDTree(locs, leaf_size=leafsize)


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
    def _retrieve_neighs_spec(self, elem_i, info_i={}, ifdistance=False):
        """Retrieve element neighbourhood information.
        """
        info_i = self._format_info_i_reg(info_i, elem_i)
        neighs, dists = self._retrieve_neighs_spec2(elem_i, **info_i)
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

    def _define_retriever(self, main_mapper):
        "Define the main mapper as a special retriever."
        self.retriever = main_mapper

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        check = True
        return check

    def _format_output(self, i_locs, neighs, dists, output):
        "Format output."
        neighs, dists = self._exclude_auto(i_locs, neighs, dists)
        neighs, dists = np.array(neighs), np.array(dists)
        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
        dists = dists.reshape((len(dists), n_dim))
        neighs, dists = self._output_map[output](i_locs, (neighs, dists))
        return neighs, dists

    def _format_info_i_reg(self, info_i, i=-1):
        "TODO: match index from relations."
        if bool(info_i):
            pass
        else:
            if type(i) == np.ndarray:
                i = np.where(self.retriever.data == i)[0]
                i = i[0] if i.any() else -1
            if self._info_ret is not None:
                if type(self._info_ret) == list:
                    if i != -1:
                        info_i = self._info_ret[i]
                    else:
                        info_i = {}
                elif type(self._info_ret) == dict:
                    info_i = self._info_ret
        return info_i

#    @property
#    def data(self):
#        "In order to keep structure."
#        if self._autodata:
#            return self.retriever.data_input
#        else:
#            return None
#
#    @property
#    def _autodata(self):
#        return


class LimDistanceEleNeigh(NetworkRetriever):
    """Region Neighbourhood based on the limit distance bound.
    """
    _default_ret_val = {}

    def _retrieve_neighs_spec2(self, elem_i, lim_distance=None, maxif=True,
                               ifdistance=True):
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
        neighs, dists = self.retriever[elem_i]
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

    def _retrieve_neighs_reg(self, elem_i, i_disc=0, lim_distance=None,
                             maxif=True, ifdistance=True):
        """Retrieve the elements which are defined by the parameters of the
        inputs and the nature of this object method.
        TODEPRECATE: or for particular use.

        Parameters
        ----------
        elem_i: int or numpy.ndarray
            the elements we want to get its neighbor element.
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
            the ids of the neighbourhood elements.
        dists: numpy.ndarray
            the distances between elements.
        """
        neighs, dists = self.retriever.retrieve_neighs(elem_i)
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


class SameEleNeigh(NetworkRetriever):
    """Network retriever which returns the same element as the mapper defined
    in the retriever maps.
    """
    _default_ret_val = {}

    def _retrieve_neighs_spec2(self, elem_i):
        """Retrieve the elements which are defined by the parameters of the
        inputs and the nature of this object method.
        TODEPRECATE: or particular use.

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
        neighs, dists = self.retriever[elem_i]
        neighs = neighs.ravel()
        return neighs, dists

    def _retrieve_neighs_reg(self, elem_i, i_disc=0):
        """Retrieve the elements which are defined by the parameters of the
        inputs and the nature of this object method.
        TODEPRECATE: or particular use.

        Parameters
        ----------
        elem_i: int or numpy.ndarray
            the element we want to get its neighsbour elements.
        i_disc: int
            the discretization we want to apply.

        Returns
        -------
        neighs: numpy.ndarray
            the ids of the neighbourhood elements.
        dists: numpy.ndarray
            the distances between elements.
        """
        if type(elem_i) == np.ndarray:
            neighs, dists = np.array(elem_i), np.array([0])
        else:
            neighs = np.array(self.retriever.data[elem_i])
            dists = np.array([0]*neighs.shape[0])
        neighs = neighs.ravel()
        return neighs, dists


class OrderEleNeigh(NetworkRetriever):
    """Network retriever based on the order it is away from element
    direct neighbours in a network.
    """

    exactorlimit = False
    _default_ret_val = {}

    def _retrieve_neighs_spec2(self, elem_i, order=0, exactorlimit=False):
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
        to_reg, to_dists = [elem_i], [self.retriever._inv_null_value]
        reg_explored = []

        ## 1. Crawling in network
        for o in range(order+1):
            neighs_o, dists_o = [], []
            for i in range(len(to_reg)):
                to_reg_i = np.array(to_reg[i])
                neighs_oi, dists_oi = self.retriever[to_reg_i]
                if self.retriever._distanceorweighs:
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

    def _retrieve_neighs_reg(self, elem_i, i_disc=0, order=0,
                             exactorlimit=False):
        """Retrieve the elements which are defined by the parameters of the
        inputs and the nature of this object method.
        TODEPRECATE: and suspicious of bad results.

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
        elem_i = self.retriever.data[elem_i] if type(elem_i) == int else elem_i
        ## 0. Needed variables
        neighs, dists, to_reg = [], [], [elem_i]
        ## Crawling in net variables
        to_reg, to_dists = [elem_i], [self.retriever._inv_null_value]
        reg_explored = []

        ## 1. Crawling in network
        for o in range(order+1):
            neighs_o, dists_o = [], []
            for i in range(len(to_reg)):
                to_reg_i = np.array(to_reg[i])
                neighs_o, dists_o = self.retriever.retrieve_neighs(to_reg_i)
                if self.retriever._distanceorweighs:
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
