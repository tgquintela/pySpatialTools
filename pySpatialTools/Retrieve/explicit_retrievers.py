
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
                 autoexclude=True, ifdistance=True, info_f=None,
                 perturbations=None, relative_pos=None, input_map=None,
                 output_map=None, constant_info=False, bool_input_idx=None):
        "Creation a element network retriever class method."
        # Reset globals
        self._initialization()
        # IO mappers
        self._format_maps(input_map, output_map)
        # Perturbations
        self._format_perturbation(perturbations)
        ## Retrieve information
        self._define_retriever(main_mapper)
        ## Info_ret mangement
        self._format_retriever_info(info_ret, info_f, constant_info)
        # Output information
        self._format_output_information(autoexclude, ifdistance, relative_pos)
        self._format_exclude(bool_input_idx, self.constant_neighs)
        ## Format retriever function
        self._format_retriever_function()
        self._format_getters(bool_input_idx)
        # Preparation input and output
        self._format_preparators(bool_input_idx)
        self._format_neighs_info(bool_input_idx)
        ## Assert properly formatted
        self.assert_correctness()

    ############################## Main functions #############################
    ###########################################################################
    def _retrieve_neighs_general_spec(self, elem_i, info_i={},
                                      ifdistance=True, kr=0):
        """Retrieve element neighbourhood information. """
        elem_i = self._prepare_input(elem_i, kr)
        print '.'*25, elem_i, self._prepare_input
        info_i = self._format_info_i_reg(info_i, elem_i)
        neighs, dists = self._retrieve_neighs_spec2(elem_i, kr=kr, **info_i)
        if ifdistance:
            neighs, dists =\
                self._apply_relative_pos_spec((neighs, dists), elem_i)
        else:
            dists = None
        return neighs, dists

    def _retrieve_neighs_constant_nodistance(self, elem_i, info_i={}, kr=0):
        """Retrieve neighs not computing distance by default.

        Parameters
        ----------
        elem_i: int
            the indice of the elem_i.
        """
        info_i = self._get_info_i(elem_i, info_i)
        elem_i = self._prepare_input(elem_i, kr)
        neighs, dists = self._retrieve_neighs_spec2(elem_i, kr=kr, **info_i)
        dists = None
        return neighs, dists

    def _retrieve_neighs_constant_distance(self, elem_i, info_i={}, kr=0):
        """Retrieve neighs not computing distance by default.

        Parameters
        ----------
        elem_i: int
            the indice of the elem_i.
        """
        info_i = self._get_info_i(elem_i, info_i)
        elem_i = self._prepare_input(elem_i, kr)
        neighs, dists = self._retrieve_neighs_spec2(elem_i, kr=kr, **info_i)
        neighs, dists = self._apply_relative_pos_spec((neighs, dists), elem_i)
        return neighs, dists

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def _define_retriever(self, main_mapper, pars_ret={}):
        """Define the main mapper as a special retriever."""
        ## TODO: Ensure correct class
        self.retriever.append(main_mapper)
        ## TODO: Compute constant neighs
        self.constant_neighs = False
        if main_mapper._input == 'indices':
            self.preferable_input_idx = True
        elif main_mapper._input == 'elements_id':
            self.preferable_input_idx = False
        else:
            self.preferable_input_idx = None

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        check = True
        return check

    def _format_output_exclude(self, i_locs, neighs, dists, output=0, kr=0):
        "Format output."
        print 'this is the point of debug', neighs, dists, i_locs
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, kr)
#        neighs, dists = np.array(neighs), np.array(dists)
#        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
#        dists = dists.reshape((len(dists), n_dim))
        print 'c'*20, neighs, dists, type(neighs)
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        print 'd'*20, neighs, dists, self.neighs_info.set_neighs, type(neighs), self._exclude_auto, self._output_map
        return neighs, dists

    def _format_output_noexclude(self, i_locs, neighs, dists, output=0, kr=0):
        "Format output."
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, kr)
#        neighs, dists = np.array(neighs), np.array(dists)
#        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
#        dists = dists.reshape((len(dists), n_dim))
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

    def _preformat_neighs_info(self, format_level=None, type_neighs=None,
                               type_sp_rel_pos=None):
        """Over-writtable function."""
        format_level, type_neighs, type_sp_rel_pos = 2, 'list', 'list'
        return format_level, type_neighs, type_sp_rel_pos

    def _get_idx_from_loc(self, loc_i, kr=0):
        """Get indices from stored data."""
        print '0'*25, self.retriever[kr].data_input
        i_loc = np.where(self.retriever[kr].data_input == loc_i)[0]
        return i_loc

    @property
    def data_input(self):
        ## Assumption kr=0 is the leading data input.
        return self.retriever[0].data_input


class LimDistanceEleNeigh(NetworkRetriever):
    """Region Neighbourhood based on the limit distance bound.
    """
    _default_ret_val = {}
    ## Basic information of the core retriever
#    preferable_input_idx = True
#    constant_neighs = False
    auto_excluded = False
    ## Interaction with the stored data
    bool_listind = False

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
        neighs: list of arrays
            the ids of the neighbourhood elements.
        dists: numpy.ndarray
            the distances between elements.

        """
        neighs, dists = self.retriever[kr][elem_i]
        for i in range(len(elem_i)):
            if np.any(neighs[i]):
                if lim_distance is None:
                    logi = np.ones(len(dists[i])).astype(bool)
                else:
                    if maxif:
                        logi = dists[i].ravel() < lim_distance
                    else:
                        logi = dists[i].ravel() > lim_distance
                print neighs, dists, logi
                neighs[i] = np.array(neighs[i][logi])
                dists[i] = np.array(dists[i][logi])
        return neighs, dists


class SameEleNeigh(NetworkRetriever):
    """Network retriever which returns the same element as the mapper defined
    in the retriever maps.
    """
    _default_ret_val = {}
    ## Basic information of the core retriever
#    preferable_input_idx = True
#    constant_neighs = False
    auto_excluded = False
    ## Interaction with the stored data
    bool_listind = False

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
        print 'k'*50, self.neighs_info.set_neighs, elem_i
        neighs, dists = self.retriever[kr][elem_i]
        print 'o'*50, neighs, type(neighs), dists
        assert(len(neighs) == len(elem_i))
        assert(all([len(e.shape) == 2 for e in dists]))
        return neighs, dists


class OrderEleNeigh(NetworkRetriever):
    """Network retriever based on the order it is away from element
    direct neighbours in a network.
    """
    _default_ret_val = {}
    ## Basic information of the core retriever
#    preferable_input_idx = True
#    constant_neighs = False
    auto_excluded = False
    ## Interaction with the stored data
    bool_listind = False

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
        neighs, dists = [[]]*len(elem_i), [[]]*len(elem_i)
        # Crawling in net variables
        to_reg = [elem_i]
        to_dists = [self.retriever[kr]._inv_null_value]*len(elem_i)

        # Order 0
        neighs_oi, dists_oi = self.retriever[kr][elem_i]
        print dists_oi, type(dists_oi), len(dists_oi), neighs_oi
        to_reg = neighs_oi
        for iss_i in range(len(elem_i)):
            if len(neighs_oi[iss_i]) != 0:
                neighs[iss_i].extend(neighs_oi[iss_i])
                dists[iss_i].extend(dists_oi[iss_i])
                to_dists[iss_i] += dists_oi[iss_i]
        print to_dists, dists, neighs

        ## 1. Crawling in network
        for o in range(1, order+1):
            for iss_i in range(len(elem_i)):
                if not np.any(to_reg[iss_i]):
                    continue
                ## Get neighbours from to_reg[iss_i]
                neighs_oi, dists_oi = self.retriever[kr][to_reg[iss_i]]
                print '9'*15, len(to_reg), len(dists_oi), len(neighs_oi)
                print iss_i, to_reg, dists_oi, neighs_oi
                ## If exact
                if o == order and exactorlimit:
                    neighs[iss_i] = neighs_oi
                    if self.retriever[kr]._distanceorweighs:
                        # Dists aggregation
                        for j in range(len(neighs_oi)):
                            dists_oi[j] += dists[iss_i][j]
                    else:
                        # Dists aggregation
                        for j in range(len(neighs_oi)):
                            dists_oi[j] += 1
                    dists[iss_i] = dists_oi

                ## Distance or weights (order)
                if self.retriever[kr]._distanceorweighs:
                    ## Filter previsited
                    new_to_reg, new_to_dists = [], []
                    i_not_visited, i_previsited, j_previsited = [], [], []
                    for i in range(len(neighs_oi)):
                        i_not_visited_i, i_previsited_i = [], []
                        for k in range(len(neighs_oi[i])):
                            if neighs_oi[i][k] not in neighs[iss_i]:
                                i_not_visited_i.append(k)
                            else:
                                i_previsited_i.append(k)
                        new_to_reg.append(neighs_oi[i][i_not_visited_i])
                        new_to_dists.append(dists_oi[i][i_not_visited_i])
                        i_not_visited.append(i_not_visited_i)
                        i_previsited.append(i_previsited_i)
                    ## Aggregate distances
                    new_dists_oi = []
                    for i in range(len(dists_oi)):
                        if len(dists_oi[i].ravel()):
                            aux_dists = to_dists[iss_i][i] + dists_oi[i]
                            new_dists_oi.append(aux_dists)
                    if len(new_dists_oi) == 0:
                        dists_oi = np.array([[]]).T
                    else:
                        dists_oi = np.concatenate(new_dists_oi)
                    assert(len(dists_oi.shape) == 2)
#                    dists_oi = np.concatenate(dists_oi)
#                    dists_oi = np.concatenate([to_dists[iss_i][i] + dists_oi[i]
#                                               for i in range(len(dists_oi))])
                    neighs_oi = np.concatenate(neighs_oi)
                    assert(len(dists_oi) == len(neighs_oi))
                    ## Internal filtering
                    u_elements = np.unique(neighs_oi)
                    new_dists, new_neighs = [], []
                    for i in range(len(u_elements)):
                        indices = list(np.where(u_elements[i] == neighs_oi)[0])
                        j_min = np.argmin(dists_oi[indices])
                        if u_elements[i] in neighs[iss_i]:
                            jm = np.where(u_elements[i] == neighs[iss_i])[0][0]
                            if dists[iss_i][jm] > dists_oi[indices[j_min]]:
                                dists[iss_i][jm] = dists_oi[indices[j_min]]
                        else:
                            new_neighs.append(neighs_oi[indices[j_min]])
                            new_dists.append(dists_oi[indices[j_min]])
                    ## Storing globals
                    neighs_oi = np.array(new_neighs)
                    if len(new_dists):
                        dists_oi = np.array(new_dists)
                    else:
                        dists_oi = np.array([[]]).T
                    print dists_oi, new_dists, dists_oi.shape
                    assert(len(dists_oi.shape) == 2)
                    neighs[iss_i] = np.concatenate([neighs[iss_i],
                                                    neighs_oi]).astype(int)
                    print dists_oi, dists[iss_i], neighs_oi, np.array(dists[iss_i]).shape, dists_oi.shape
                    dists[iss_i] = np.concatenate([np.array(dists[iss_i]),
                                                   dists_oi])
#                    dists[iss_i] = dists[iss_i].reshape((len(dists[iss_i]), 1))
                    to_reg, to_dists = new_to_reg, new_to_dists
                else:
                    ## Dists aggregation
                    for j in range(len(neighs_oi)):
                        dists_oi[j] += 1
                    ## Formatting
                    neighs_oi = np.concatenate(neighs_oi)
                    dists_oi = np.concatenate(dists_oi)
                    ## Filter previsited
                    neighs_oi, idxs = np.unique(neighs_oi, True)
                    dists_oi = dists_oi[idxs]
                    indices = [i for i in range(len(neighs_oi))
                               if neighs_oi[i] not in neighs[iss_i]]
                    ## Storing globals
                    aux = np.concatenate([neighs[iss_i], neighs_oi[indices]])
                    neighs[iss_i] = aux
                    dists[iss_i] = np.concatenate([dists[iss_i],
                                                   dists_oi[indices]])
                    to_reg[iss_i] = neighs_oi[indices]
                    to_dists[iss_i] = dists_oi[indices]
        return neighs, dists
