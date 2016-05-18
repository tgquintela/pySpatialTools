
"""
Explicit retrievers
------------------
The module which groups the explicit defined retrievers of generic elements.

"""

import numpy as np
#from itertools import combinations
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
                 autoexclude=False, ifdistance=True, info_f=None,
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
#        print '.'*25, elem_i, self._prepare_input
#        info_i = self._format_info_i_reg(info_i, elem_i)
        info_i = self._get_info_i(elem_i, info_i)
#        print '_'*25, info_i, elem_i, self._get_info_i, self._info_ret
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

    ############################# Getter functions ############################
    ###########################################################################
    def _get_loc_from_idx(self, i, kr=0):
        """Not list indexable interaction with data."""
        loc_i = np.array(self.retriever[kr].data_input[i])
        return loc_i

    def _get_idx_from_loc(self, loc_i, kr=0):
        """Get indices from stored data."""
        i_loc = np.where(self.retriever[kr].data_input == loc_i)[0]
        i_loc = list(i_loc) if len(i_loc) else []
        return i_loc

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def _define_retriever(self, main_mapper, pars_ret={}):
        """Define the main mapper as a special retriever."""
        ## TODO: Ensure correct class
        self.retriever.append(main_mapper)
        ## TODO: Compute constant neighs
        assert(main_mapper._input in ['indices', 'elements_id'])
        self.constant_neighs = False
        if main_mapper._input == 'indices':
            self.preferable_input_idx = True
        elif main_mapper._input == 'elements_id':
            self.preferable_input_idx = False
        ## Assert input output
        assert(main_mapper._out == 'indices')
#        else:
#            raise Exception("Not possible option.")
#            self.preferable_input_idx = None

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        check = True
        return check

    def _format_output_exclude(self, i_locs, neighs, dists, output=0, kr=0):
        "Format output."
#        print 'this is the point of debug', neighs, dists, i_locs
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, kr)
#        neighs, dists = np.array(neighs), np.array(dists)
#        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
#        dists = dists.reshape((len(dists), n_dim))
#        print 'c'*20, neighs, dists, type(neighs)
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
#        print 'd'*20, neighs, dists, self.neighs_info.set_neighs, type(neighs), self._exclude_auto, self._output_map
        return neighs, dists

    def _format_output_noexclude(self, i_locs, neighs, dists, output=0, kr=0):
        "Format output."
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, kr)
#        neighs, dists = np.array(neighs), np.array(dists)
#        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
#        dists = dists.reshape((len(dists), n_dim))
        return neighs, dists

    def _preformat_neighs_info(self, format_level=None, type_neighs=None,
                               type_sp_rel_pos=None):
        """Over-writtable function."""
        format_level, type_neighs, type_sp_rel_pos = 2, 'list', 'list'
        return format_level, type_neighs, type_sp_rel_pos

    @property
    def data_input(self):
        ## Assumption kr=0 is the leading data input.
        return self.retriever[0].data_input


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
#        print 'k'*50, self.neighs_info.set_neighs, elem_i
        neighs, dists = self.retriever[kr][elem_i]
#        print 'o'*50, neighs, type(neighs), dists
        assert(len(neighs) == len(elem_i))
        assert(all([len(e.shape) == 2 for e in dists]))
        return neighs, dists


class LimDistanceEleNeigh(NetworkRetriever):
    """Region Neighbourhood based on the limit distance bound.
    """
    _default_ret_val = {'lim_distance': 1}
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
#                print neighs, dists, logi
                neighs[i] = np.array(neighs[i][logi])
                dists[i] = np.array(dists[i][logi])
        return neighs, dists


class OrderEleNeigh(NetworkRetriever):
    """Network retriever based on the order it is away from element
    direct neighbours in a network.
    """
    _default_ret_val = {'order': 0, 'exactorlimit': False}
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
        order = 2
        ## 0. Needed variables
        neighs, dists = [[]]*len(elem_i), [[]]*len(elem_i)
        if self.retriever[kr]._distanceorweighs:
            nullvalue = self.retriever[kr]._null_value
        else:
            nullvalue = 0
        ## Searching extensively (storing repeated results)
        extensive = exactorlimit or self.retriever[kr]._distanceorweighs
        # Crawling in net variables
        to_reg = [elem_i]
        to_dists = [[nullvalue]]*len(elem_i)

        # Order 0
#        print '0'+'.'*50
#        print elem_i, neighs, dists, to_reg, to_dists
        neighs_oi, dists_oi = self.retriever[kr][elem_i]
#        assert(len(neighs_oi) == len(dists_oi))
#        assert(len(elem_i) == len(neighs_oi))
#        print '1'+'.'*50
#        print dists_oi, type(dists_oi), len(dists_oi), neighs_oi
#        assert(all([len(d.shape) == 2 for d in dists_oi]))
#        assert(all([len(dists_oi[i]) == len(neighs_oi[i])
#                    for i in range(len(dists_oi))]))
        if order > 0:
            to_reg = list(neighs_oi)
            for iss_i in range(len(elem_i)):
                if len(neighs_oi[iss_i]) != 0:
                    neighs[iss_i].extend(neighs_oi[iss_i])
                    dists[iss_i].extend(dists_oi[iss_i])
                    to_dists[iss_i] += dists_oi[iss_i]
#            print '2'+'.'*50
#            print to_reg, to_dists, dists, neighs
#            assert(len(to_dists) == len(to_reg))
#            assert(len(to_reg) == len(elem_i))
#            assert(len(neighs) == len(dists))
#            assert(all([len(neighs[i]) == len(dists[i])
#                        for i in range(len(neighs))]))
#            assert(all([type(nei) == list for nei in neighs]))
#            assert(all([type(d) == list for d in dists]))

        ## 1. Crawling in network
        for o in range(2, order+1):
            for iss_i in range(len(elem_i)):
                ## Stop crawling this iss_i if there is no toreg
                if not np.any(to_reg[iss_i]):
                    continue
                ## Get neighbours from to_reg[iss_i]
                neighs_oi, dists_oi = self.retriever[kr][to_reg[iss_i]]
#                assert(len(neighs_oi) == len(dists_oi))
                ## Adding distances
                dists_oi = _adding_dists(self.retriever[kr]._distanceorweighs,
                                         dists_oi, to_dists[iss_i])
                ## Concatenation
                if len(neighs_oi):
                    neighs_oi = list(np.concatenate(neighs_oi).astype(int))
                    dists_oi = list(np.concatenate(dists_oi))

                ## If exact and order reach
                if o == order and exactorlimit:
                    neighs[iss_i] = neighs_oi
                    dists[iss_i] = dists_oi
                    continue

                ## Filtering if not extensive
                if not extensive:
                    neighs_oi, dists_oi = _filtering_extensive(iss_i, neighs,
                                                               neighs_oi,
                                                               dists_oi)
                ## Storing
                if np.any(neighs_oi):
                    neighs[iss_i].extend(neighs_oi)
                    dists[iss_i].extend(dists_oi)
                to_reg[iss_i] = neighs_oi
                to_dists[iss_i] = dists_oi
        ## Selecting unique
        if self.retriever[kr]._distanceorweighs:
            neighs, dists = _unique_less_dists(neighs, dists)
        return neighs, dists


###############################################################################
############################# Auxiliar functions ##############################
###############################################################################
######################### Order retriever auxiliars ###########################
numbertypes = [int, float, np.float, np.int32, np.int64]
arraytypes = [np.ndarray, list]


def _filtering_extensive(iss, neighs, neighs_oi, dists_oi):
    """Filtering previously visited nodes."""
    indices = [i for i in range(len(neighs_oi))
               if neighs_oi[i] not in neighs[iss]]
    neighs_oi = [neighs_oi[i] for i in indices]
    dists_oi = [dists_oi[i] for i in indices]
    return neighs_oi, dists_oi


def _adding_dists(_distanceorweighs, dists_oi, to_dists_oi):
    """Adding predistances to new distances."""
#    print _distanceorweighs, dists_oi, to_dists_oi
#    assert(_distanceorweighs in [True, False])
#    assert(len(dists_oi) == len(to_dists_oi))
#    assert(all([type(e) in numbertypes for e in to_dists_oi]))
#    assert(all([type(e) in arraytypes for e in dists_oi]))
    if _distanceorweighs:
        for j in range(len(dists_oi)):
            dists_oi[j] += to_dists_oi[j]
    else:
        # Dists aggregation
        for j in range(len(dists_oi)):
            dists_oi[j] += 1
    return dists_oi


def _unique_less_dists(neighs, dists):
    """Selection of less dists neighs."""
    for iss_i in range(len(neighs)):
        uniques = list(np.unique(neighs[iss_i]))
        neighs_i = np.array(neighs[iss_i]).astype(int)
        dists_i = np.array(dists[iss_i]).ravel()
        uniques_dists = []
        for j in range(len(uniques)):
            aux_dists = np.array([np.min(dists_i[uniques[j] == neighs_i])])
            uniques_dists.append(aux_dists)
        neighs[iss_i] = uniques
        dists[iss_i] = uniques_dists
    return neighs, dists
