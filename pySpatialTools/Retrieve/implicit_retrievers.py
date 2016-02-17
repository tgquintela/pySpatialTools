
"""
Implicit retrievers
-------------------
Implicit defined retrievers grouped in this module.

"""

import numpy as np
from sklearn.neighbors import KDTree
from retrievers import Retriever


###############################################################################
############################# Element Retrievers ##############################
###############################################################################
class ElementRetriever(Retriever):
    """Retreiver of elements given other elements and only considering the
    information of the non-retrivable elements.
    """
    typeret = 'element'

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
        ## Reset globals
        self._initialization()
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

    def _format_output(self, i_locs, neighs, dists, output, k=0):
        "Format output."
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, k)
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

    def _retrieve_neighs_spec(self, point_i, kneighs_i, ifdistance=False,
                              kr=0):
        "Function to retrieve neighs in the specific way we want."
        point_i = self._get_loc_i(point_i)
        res = self.retriever[kr].query(point_i, int(kneighs_i), ifdistance)
        if ifdistance:
            res = res[1][0], res[0][0]
        else:
            res = res[0], [[] for i in range(len(res[0]))]
        ## Correct for another relative spatial measure
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever[kr].data)[res[0], :]
            if type(self.relative_pos).__name__ == 'function':
                res = res[0], self.relative_pos(point_i, loc_neighs)
            else:
                res = res[0], self.relative_pos.compute(point_i, loc_neighs)
        return res

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        try:
            for k in range(self.k_perturb):
                _, kr = self._map_perturb(k)
                loc = self.retriever[kr].data[0]
                self.retriever[kr].query(loc, 2)
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
        retriever = KDTree(locs, leaf_size=leafsize)
        self.retriever.append(retriever)


################################ R disctance ##################################
###############################################################################
class CircRetriever(SpaceRetriever):
    "Circular retriever."
    _default_ret_val = 0.1

    def _retrieve_neighs_spec(self, point_i, radius_i, ifdistance=False, kr=0):
        "Function to retrieve neighs in the specific way we want."
        point_i = self._get_loc_i(point_i)
        res = self.retriever[kr].query_radius(point_i, radius_i, ifdistance)
        if ifdistance:
            res = res[0][0], res[1][0]
        else:
            res = res[0], [[] for i in range(len(res[0]))]
        ## Correct for another relative spatial measure
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever[kr].data)[res[0], :]
            if type(self.relative_pos).__name__ == 'function':
                res = res[0], self.relative_pos(point_i, loc_neighs)
            else:
                res = res[0], self.relative_pos.compute(point_i, loc_neighs)
        return res

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        try:
            for k in range(self.k_perturb):
                _, kr = self._map_perturb(k)
                loc = self.retriever[kr].data[0]
                self.retriever[kr].query_radius(loc, 0.5)
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
        retriever = KDTree(locs, leaf_size=leafsize)
        self.retriever.append(retriever)
