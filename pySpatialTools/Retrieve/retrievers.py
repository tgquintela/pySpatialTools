
"""
Retrievers
----------
The objects to retrieve neighbours in flat (non-splitted) space or precomputed
mapped relations.


Structure:
----------


TODO:
----
- Ifdistance better implementation
- Exclude better implementation
- Multiple regions
- Multiple points to get neighs

"""

import numpy as np
import warnings
from scipy.sparse import coo_matrix
from ..utils import NonePerturbation
from ..utils import ret_filter_perturbations
from ..utils.util_classes import SpatialElementsCollection


class Retriever:
    """Class which contains the retriever of points.
    """
    __name__ = 'pySpatialTools.Retriever'

    def set_locs(self, locs, info_ret, info_f):
        "Set locations for retrieving their neighs."
        self.data = SpatialElementsCollection(locs)
        self._autodata = False
        self._format_retriever_info(info_ret, info_f)

    def retrieve_neighs(self, i_loc, info_i={}, ifdistance=None, k=None,
                        output=0):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        """
        ## 0. Prepare variables
        # Prepare information retrieve
        info_i = self._get_info_i(i_loc, info_i)
        ifdistance = self._ifdistance if ifdistance is None else ifdistance
        # Prepare perturbation index
        ks = [k] if type(k) == int else k
        ks = 0 if ks is None else ks
        ## 1. Retrieve neighs
        if ks == 0 or self.staticneighs:
            # Prepare coordinates information (depending on coord or index)
            try:
                i_mloc = self._input_map(self, i_loc)
            except:
                loc = np.array(self.retriever[0].data)[i_loc]
                i_mloc = self._input_map(self, loc)
            # Get neighs info
            neighs, dists =\
                self._retrieve_neighs_spec(i_mloc, info_i, ifdistance)
            ## 2. Exclude auto if it is needed
            neighs_info = self._format_output(i_loc, neighs, dists, output)
        else:
            neighs_info = []
            for k in range(len(ks)):
                # Prepare coordinates information (depending on coord or index)
                try:
                    i_mloc = self._input_map(self, i_loc)
                except:
                    loc = np.array(self.retriever[k].data)[i_loc]
                    i_mloc = self._input_map(self, loc)
                # Map perturb
                _, k_r = self._map_perturb(k)
                # Retrieve
                neighs, dists =\
                    self._retrieve_neighs_spec(i_mloc, info_i, ifdistance, k_r)
                nei_k = self._format_output(i_loc, neighs, dists, output, k_r)
                neighs_info.append(nei_k)
        return neighs_info

    ######################### Perturbation management #########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Add perturbations."""
        perturbations = ret_filter_perturbations(perturbations)
        if type(perturbations) == list:
            for p in perturbations:
                self._dim_perturb.append(p.k_perturb)
                self._perturbators.append(p)
                self._create_map_perturbation()
                self._add_perturbated_retrievers(p)
        else:
            self._dim_perturb.append(perturbations.k_perturb)
            self._perturbators.append(perturbations)
            self._create_map_perturbation()
            self._add_perturbated_retrievers(perturbations)

    def _format_perturbation(self, perturbations):
        "Format initial perturbations."
        if perturbations is None:
            def _map_perturb(x):
                if x != 0:
                    raise IndexError("Not perturbation available.")
                return 0, 0
            self._map_perturb = _map_perturb
            self._dim_perturb = [1]
        else:
            self.add_perturbations(perturbations)

    def _create_map_perturbation(self):
        """Create the map for getting the perturbation object.
        The first inde of the map_perturb is an index in the _perturbators,
        the other one is an index for the retrievers.
        """
        ## 0. Creation of the mapper array
        limits = np.cumsum([0] + list(self._dim_perturb))
        sl = [slice(limits[i], limits[i+1]) for i in range(len(limits)-1)]
        ## Build a mapper
        mapper = np.zeros((np.sum(self._dim_perturb), 2)).astype(int)
        for i in range(len(sl)):
            inds = np.zeros((sl[i].stop-sl[i].start, 2))
            inds[:, 0] = i
            if self._perturbators[i]._perturbtype != 'none':
                ## TODO: perturbations of aggregations (networks)
                if self.typeret == 'space':
                    max_n = mapper[:, 1].max() + 1
                    inds[:, 1] = np.arange(sl[i].stop-sl[i].start) + max_n
            mapper[sl[i]] = inds

        ## 1. Creation of the mapper function
        def map_perturb(x):
            if x < 0:
                raise IndexError("Negative numbers can not be indices.")
            if x > self.k_perturb:
                msg = "Out of bounds. There are only %s perturbations."
                raise IndexError(msg % str(self.k_perturb))
            return mapper[x]
        ## 2. Storing mapper function
        self._map_perturb = map_perturb

    def _add_perturbated_retrievers(self, perturbation):
        """Add a perturbated retriever in self.retriever using the class
        function self._define_retriever."""
        if perturbation._categorytype == 'location':
            self.staticneighs = False
            for k in range(perturbation.k_perturb):
                locs_p = perturbation.apply2locs(self.retriever[0].data, k=k)
                self._define_retriever(locs_p[:, :, 0])

    @property
    def k_perturb(self):
        return np.sum(self._dim_perturb)-1

    ######################### Aggregation management ##########################
    ###########################################################################

    ########################### Auxiliar functions ############################
    ###########################################################################
    def _initialization(self):
        """Mutable globals reset."""
        ## Elements information
        self.data = None
        self._autodata = False
        ## Retriever information
        self.retriever = []
        self._info_ret = None
        self._info_f = None
        ## Perturbation
        self._dim_perturb = [1]
        self._map_perturb = lambda x: (0, 0)
        self._perturbators = [NonePerturbation()]
        self.staticneighs = True
        ## External objects to apply
        self.relative_pos = None
        ## IO information
        self._flag_auto = False
        self._ifdistance = False
        self._autoret = False
        self._heterogenous_input = False
        self._heterogenous_output = False
        ## IO methods
        self._input_map = lambda s, i: i
        self._output_map = [lambda s, i, x: x]

    def _format_retriever_info(self, info_ret, info_f):
        "Format properly the retriever information."
        if type(info_ret).__name__ == 'function':
            self._info_f = info_ret
        else:
            self._info_f = info_f
        aux_default = self._default_ret_val
        self._info_ret = aux_default if info_ret is None else info_ret

    def _exclude_auto(self, i_loc, neighs, dists, k=0):
        "Exclude auto points if there exist in the neighs retrieved."
        ## 0. Detect input i_loc and retrieve to_exclude_points list
        if type(i_loc) in [int, np.int32, np.int64]:
            to_exclude_points = [i_loc]
        elif type(i_loc) == np.ndarray:
            ###########################################################
            to_exclude_points = self._build_excluded_points(i_loc, k)
            ###########################################################
        ## 1. Excluding task
        n_p = np.array(neighs).shape[0]
        idxs_exclude = [i for i in xrange(n_p) if neighs[i]
                        in to_exclude_points]
        neighs = [neighs[i] for i in xrange(n_p) if i not in idxs_exclude]
        if dists is not None:
            dists = [dists[i] for i in xrange(n_p) if i not in idxs_exclude]
        return neighs, dists

    def _build_excluded_points(self, i_loc, k=0):
        "Build the excluded points from i_loc."
        sh = i_loc.shape
        i_loc = i_loc if len(sh) == 2 else i_loc.reshape(1, sh[0])
        try:
            logi = np.all(self.retriever[k].data == i_loc, axis=1).ravel()
        except:
            logi = np.all(self.retriever[k].data == i_loc)
        assert len(logi) == len(self.retriever[k].data)
#        logi = np.ones(len(self.retriever.data)).astype(bool)
#        for i in range(self.retriever.data.shape[1]):
#            aux_logi = np.array(self.retriever.data)[:, i] == i_loc[:, i]
#            logi = np.logical_and(logi, aux_logi)
        to_exclude_points = np.where(logi)[0]
        return to_exclude_points

    def _check_coord(self, i_locs, k=0, inorout=True):
        """Function to check if the input are coordinates or indices. The input
        is a coordinate when is an array with the same dimension that the pool
        of retrievable locations stored in retriever.data or in self.data.

        Parameters
        ----------
        i_locs: int, list of ints, numpy.ndarray or list of numpy.ndarray
            the locations information.

        Returns
        -------
        checker_coord: boolean
            if there are coordinates True, if there are indices False.

        """
        ## Get individuals
        if type(i_locs) == list:
            check_loc = i_locs[0]
        else:
            check_loc = i_locs
        ## Get checker
        if type(check_loc) in [int, np.int32, np.int64, np.ndarray]:
            if type(check_loc) != np.ndarray:
                checker_coord = False
            else:
                flag = inorout and not self._autoret
                if flag:
                    d_sh = self.data.shape
                else:
                    d_sh = self.retriever[k].data.shape
                if len(check_loc.shape) == len(d_sh):
                    checker_coord = True
                else:
                    raise IndexError("Not correct shape for coordinates.")
        else:
            checker_coord = None
        return checker_coord

    def _get_info_i(self, i_loc, info_i):
        """Get information of retrieving point for each i_loc. Comunicate the
        input i with the data_input.
        """
        if not info_i:
            if type(i_loc) in [int, np.int32, np.int64]:
                if type(self._info_ret) in [list, np.ndarray]:
                    info_i = self._info_ret[i_loc]
                else:
                    info_i = self._info_ret
            else:
                if self._info_f is None:
                    return {}
                if type(self._info_f).__name__ == 'name':
                    info_i = self._info_f(i_loc, info_i)
                else:
                    raise TypeError("self._info_f not defined properly.")
        return info_i

    def _get_loc_i(self, i_loc, k=0, inorout=True):
        "Get location."
        ## 0. Needed variable computations
        ifdata = inorout and not self._autodata
        sh = self.data_input.shape
        if ifdata:
            flag = isinstance(self.data, SpatialElementsCollection)
        else:
            flag = isinstance(self.retriever[k].data,
                              SpatialElementsCollection)
        ## 1. Loc retriever
        if type(i_loc) in [int, np.int32, np.int64]:
            if flag:
                if ifdata:
                    i_loc = self.data_input[i_loc].location
                    i_loc = np.array(i_loc).reshape((1, sh[1]))
                else:
                    i_loc = self.data_input[i_loc].location
                    i_loc = np.array(i_loc).reshape((1, sh[1]))
            else:
                if ifdata:
                    loc_i = np.array(self.data[i_loc]).reshape((1, sh[1]))
                else:
                    loc_i = np.array(self.retriever[k].data[i_loc])
                    loc_i = loc_i.reshape((1, sh[1]))
        elif type(i_loc) in [list, np.ndarray]:
            loc_i = np.array(loc_i).reshape((1, sh[1]))
        elif isinstance(i_loc, SpatialElementsCollection):
            i_loc = np.array(i_loc.location).reshape((1, sh[1]))
        return loc_i

    ###########################################################################
    ########################### Auxiliary functions ###########################
    ###########################################################################
    def __getitem__(self, i):
        "Perform the map assignation of the neighbourhood."
        neighs, dists = self.retrieve_neighs(i)
        return neighs, dists

    @property
    def _n0(self):
        if self._heterogenous_input:
            raise Exception("Impossible action. Heterogenous input.")
        n0 = len(self.data_input)
        return n0

    @property
    def _n1(self):
        if self._heterogenous_output:
            raise Exception("Impossible action. Heterogenous output.")
        n1 = len(self.data_output)
        return n1

    @property
    def data_input(self):
        if self._autodata:
            return np.array(self.retriever[0].data)
        else:
            if self.data is None:
                self._autodata = True
                return np.array(self.data_input)
            else:
                return np.array(self.data)

    @property
    def data_output(self):
        return np.array(self.retriever[0].data)

    def compute_neighnet(self, out='sparse'):
        """Compute the relations neighbours and build a network or multiplex
        with the defined retriever class"""
        ## 0. Conditions to ensure
        if self._heterogenous_output:
            msg = "Dangerous action. Heterogenous output."
            msg += "Only will be considered the 0 output_map"
            warnings.warn(msg)
        ## 00. Define global variables
        neighs, dists = self[0]
        try:
            n_data = np.array(dists).shape[1]
        except:
            n_data = 1
        sh = (self._n0, self._n1)
        ## 1. Computation
        iss, jss = [], []
        data = [[] for i in range(n_data)]
        for i in xrange(self._n0):
            neighs, dists = self[i]
            #dists = np.array(dists).reshape((len(dists), n_data))
            n_i = len(neighs)
            if n_i != 0:
                iss_i, jss_i = [i]*n_i, list(neighs)
                iss.append(iss_i)
                jss.append(jss_i)
                for k in range(n_data):
                    data[k] += dists
        ## 2. Format output
        iss, jss = np.hstack(iss), np.hstack(jss)
        data = [np.hstack(data[k]) for k in range(n_data)]
        nets = []
        for k in range(n_data):
            nets.append(coo_matrix((data[k], (iss, jss)), shape=sh))
        #nets = [format_relationship(nets[k], out) for k in range(n_data)]
        if n_data == 1:
            nets = nets[0]
        return nets
