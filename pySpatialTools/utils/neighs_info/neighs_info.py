
"""
neighs information
------------------
Auxiliar class in order to manage the information of the neighbourhood
returned by the retrievers.
Due to the complexity of the structure it is convenient to put altogether
in a single class and manage in a centralized way all the different
interactions with neighs_info in the whole package.


possible inputs
---------------
* integer {neighs}
* list of integers {neighs}
* list of lists of integers {neighs for some iss}
* list of lists of lists of integers {neighs for some iss and ks}
* numpy array 1d, 2d, 3d {neighs}
* tuple of neighs


standart storing
----------------
- neighs:
    - array 3d (ks, iss, neighs)
    - lists [ks][iss][neighs]
    - list arrays: [ks](iss, neighs), [ks][iss](neighs)

- sp_relative_pos:
    - array 3d (ks, iss, neighs)
    - lists [ks][iss][neighs]
    - list arrays [ks](iss, neighs), [ks][iss](neighs)

standart output
---------------
- neighs:
    - array 3d (ks, iss, neighs)
    - lists [ks][iss][neighs]
    - list arrays: [ks](iss, neighs), [ks][iss](neighs)


Parameters
----------
staticneighs: all the ks have the same information. They are static.
    It is useful information for the getters. The information is stored with
    deep=2.
staticneighs_set: all the same information but it is setted as if there was
    set with deep=3. If True, deep=2, if False, deep=3.
constant_neighs: all the iss have the same number of neighs for all ks.
level: the format level expected. First one is only neighs, second one has
    different iss and the third one different ks.
_kret: maximum number of perturbations of the system. It could be useful for
    open systems expressed in a staticneighs way to find errors or delimitate
    ouptut.
n: maximum number of id of elements retrieved.

"""

import numpy as np
from copy import deepcopy
import warnings
warnings.filterwarnings("always")

from auxiliar_joinning_neighs import join_neighsinfo_AND_static_dist,\
    join_neighsinfo_OR_static_dist, join_neighsinfo_XOR_static_dist,\
    join_neighsinfo_AND_static_notdist, join_neighsinfo_OR_static_notdist,\
    join_neighsinfo_XOR_static_notdist, join_neighsinfo_AND_notstatic_dist,\
    join_neighsinfo_OR_notstatic_dist, join_neighsinfo_XOR_notstatic_dist,\
    join_neighsinfo_AND_notstatic_notdist,\
    join_neighsinfo_OR_notstatic_notdist, join_neighsinfo_XOR_notstatic_notdist

pos_structure = [None, 'raw', 'tuple', 'tuple_only', 'tuple_tuple',
                 'list_tuple_only', 'tuple_list_tuple']
pos_levels = [None, 0, 1, 2, 3]
pos_format_set_iss = [None, "general", "null", "int", "list"]
pos_types_neighs = [None, "general", "list", "array", "slice"]
pos_types_rel_pos = [None, "general", "list", "array"]
inttypes = [int, np.int32, np.int64]


class Neighs_Info:
    """Class to store, move and manage the neighbourhood information retrieved.
    """
    type_ = "pySpatialTools.Neighs_Info"

    def __init__(self, constant_neighs=False, kret=1, format_structure=None,
                 n=0, format_get_info=None, format_get_k_info=None,
                 format_set_iss=None, staticneighs=None, ifdistance=None,
                 type_neighs=None, type_sp_rel_pos=None, format_level=None):
        """The instanciation of the container object for all the neighbourhood
        information.

        Parameters
        ----------
        constant_neighs: boolean (default=False)
            if there are always the same number of neighs across all the
            possible neighs.
        kret: int (default=1)
            the total perturbations applied (maximum k size).
        format_structure: str, optional (default=None)
            the type of structure in which we are going to set the
            neighbourhood information.
        n: int (default=0)
            the maximum number of possible neighs code.
        format_get_info: str optional (default=None)
            in which format the information is returned to the user.
        format_get_k_info: str optional (default=None)
            in which format of the ks we set.
        format_set_iss: str optional (default=None)
            in which format of elements iss we set.
        staticneighs: boolean (default=None)
            if there is constant neighbourhood across the perturbations.
        ifdistance: boolean (default=None)
            if we set the distance or the relative position information.
        type_neighs: str optional (default=None)
            the type of object describing the neighs of the neighbourhood.
        type_sp_rel_pos: str optional (default=None)
            the type of object describing the relative position of the
            neighbourhood.
        format_level: int (default=None)
            the level in which the information of the neighborhood will be set.

        """
        ## Initialize class
        self._set_init()
        ## Extra info
        self._constant_neighs = constant_neighs
        # Constrain information
        self._kret = kret
        self._n = n
        # Setting and formatting information
        self.format_set_info = format_structure, type_neighs, type_sp_rel_pos,\
            format_set_iss
        self.format_get_info = format_get_info, format_get_k_info
        ## Formatters
        # Global information
        self._format_globalpars(staticneighs, ifdistance, format_level)
        # Format setters
        self._format_setters(format_structure, type_neighs,
                             type_sp_rel_pos, format_set_iss)
        # Format getters
        self._format_getters(format_get_info, format_get_k_info)
        # Format joining
        self._format_joining_functions()

    def __iter__(self):
        """Get information sequentially.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i`.
        sp_relpos: list or np.ndarray
            the relative position information for each element `i`.
        ks: list or np.ndarray
            the perturbations indices associated with the returned information.
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        for i in range(len(self.ks)):
            yield self.get_neighs([i]), self.get_sp_rel_pos([i]),\
                [self.ks[i]], self.iss

    def empty(self):
        """If it is empty."""
        return not self.any()

    def any(self):
        """If it is not empty."""
        boolean = True
        if type(self.idxs) == np.ndarray:
            boolean = all(self.idxs.shape)
        elif type(self.idxs) == list:
            sh = np.array(self.idxs).shape
            if len(sh) >= 2:
                boolean = np.all(sh)
        return boolean

    def reset(self):
        """Reset all the class to empty all the neighbourhood information."""
        self._set_init()

    def copy(self):
        """Deep copy of the container."""
        return deepcopy(self)

    @property
    def shape(self):
        """Return the number of indices, neighbours and ks considered. For
        irregular cases the neighbours number is set as None.

        Returns
        -------
        sh0: int
            the number of elements we want to get their neighbourhood.
        sh1: int
            the number of neighs they have it is constant.
        sh2: int
            the number of perturbations applied.

        """
        if not self._setted:
            return None, None, None
        if type(self.idxs) == slice:
            sh0 = len(self.iss)
            step = self.idxs.step
            sh1 = (self.idxs.stop + step - 1 - self.idxs.start)/step
            sh1 = 0 if self.ks is None else len(self.ks)
        elif type(self.idxs) == np.ndarray:
            sh0 = 0 if self.idxs is None else len(self.idxs)
            sh1 = 0 if self.idxs is None else self.idxs.shape[1]
        elif type(self.idxs) == list:
            sh0 = len(self.idxs)
            sh1 = len(self.idxs[0])
        sh2 = len(self.ks) if self.ks is not None else None
        return sh0, sh1, sh2

    ###########################################################################
    ############################ GENERAL SETTINGS #############################
    ###########################################################################
    def set_information(self, k_perturb=0, n=0):
        """Set specific global information.

        Parameters
        ----------
        kret: int (default=0)
            the total perturbations applied (maximum k size).
        n: int (default=0)
            the maximum number of possible neighs code.

        """
        self._n = n
        self._kret = k_perturb

    def _set_ks_static(self, ks):
        """External set ks for staticneighs.

        Parameters
        ----------
        ks: list or np.ndarray
            the perturbations indices associated with the stored information.

        """
        self.ks = ks
        if np.max(self.ks) > self._kret:
            self._kret = np.max(self.ks)

    def _set_ks_dynamic(self, ks):
        """External set ks for non-staticneighs.

        Parameters
        ----------
        ks: list or np.ndarray
            the perturbations indices associated with the stored information.

        """
        assert(len(ks) == len(self.idxs))
        self.ks = ks
        if np.max(self.ks) > self._kret:
            self._kret = np.max(self.ks)

    def direct_set(self, neighs, sp_relative_pos=None):
        """Direct set of neighs_info.

        Parameters
        ----------
        neighs: list or np.ndarray
            the neighs information for each element `i` and for each
            perturbation `k`.
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        self.idxs = neighs
        self.sp_relative_pos = sp_relative_pos
        self.assert_goodness()

    def reset_functions(self):
        """Reset the function regarding the parameters set."""
        if type(self.idxs) == list:
            type_neighs = 'list'
        elif type(self.idxs) == slice:
            type_neighs = 'slice'
        elif type(self.idxs) == np.ndarray:
            type_neighs = 'array'
        if type(self.sp_relative_pos) == list:
            type_sp_rel_pos = 'list'
        elif type(self.sp_relative_pos) == np.ndarray:
            type_sp_rel_pos = 'array'
        else:
            type_sp_rel_pos = None
        self.set_types(type_neighs, type_sp_rel_pos)

    def reset_structure(self, format_structure):
        """Reset structure regarding the parameters set and the
        `format_structure` input.

        Parameters
        ----------
        format_structure: str, optional
            the type of structure in which we are going to set the
            neighbourhood information.

        """
        assert(format_structure in pos_structure)
        _, aux1, aux2, aux3 = self.format_set_info
        self.format_set_info = format_structure, aux1, aux2, aux3
        self.reset_format()

    def reset_level(self, format_level):
        """Reset level regarding the parameters set and the new input.

        Parameters
        ----------
        format_level: int
            the level in which the information of the neighborhood will be set.

        """
        assert(format_level in pos_levels)
        self.level = format_level
        self.reset_format()

    def reset_format(self):
        """Reset format regarding the parameters set."""
        ## Formatters
        self._format_setters(*self.format_set_info)
        self._format_getters(*self.format_get_info)
        self._format_joining_functions()

    def set_types(self, type_neighs=None, type_sp_rel_pos=None):
        """Set type of objects in which the information will be given.

        Parameters
        ----------
        type_neighs: str optional (default=None)
            the type of object describing the neighs of the neighbourhood.
        type_sp_rel_pos: str optional (default=None)
            the type of object describing the relative position of the
            neighbourhood.

        """
        ## 1. Set set_sp_rel_pos
        self.type_neighs, self.type_sp_rel_pos = type_neighs, type_sp_rel_pos

        if self.ifdistance is False:
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        else:
            self.get_sp_rel_pos = self._general_get_rel_pos
            if self.level < 2:
                self.get_sp_rel_pos = self._static_get_rel_pos
            if type_sp_rel_pos is None or type_sp_rel_pos == 'general':
                self.set_sp_rel_pos = self._general_set_rel_pos
            elif type_sp_rel_pos == 'array':
                if self.level is None:
                    self.set_sp_rel_pos = self._set_rel_pos_general_array
                elif self.level == 0:
                    self.set_sp_rel_pos = self._set_rel_pos_dim
                elif self.level == 1:
                    self.set_sp_rel_pos = self._array_only_set_rel_pos
                elif self.level == 2:
                    self.set_sp_rel_pos = self._array_array_set_rel_pos
                elif self.level == 3:
                    self.set_sp_rel_pos = self._array_array_array_set_rel_pos
            elif type_sp_rel_pos == 'list':
                if self.level is None:
                    self.set_sp_rel_pos = self._set_rel_pos_general_list
                elif self.level == 0:
                    self.set_sp_rel_pos = self._set_rel_pos_dim
                elif self.level == 1:
                    self.set_sp_rel_pos = self._list_only_set_rel_pos
                elif self.level == 2:
                    self.set_sp_rel_pos = self._list_list_only_set_rel_pos
                elif self.level == 3:
                    self.set_sp_rel_pos = self._list_list_set_rel_pos

        ## 2. Set set_neighs
        if type_neighs is None or type_neighs == 'general':
            self.set_neighs = self._general_set_neighs
        elif type_neighs == 'array':
            # Format get neighs
            if self.staticneighs:
                self.get_neighs = self._get_neighs_array_static
            else:
                self.get_neighs = self._get_neighs_array_dynamic
            # Format set neighs
            if self.level is None:
                self.set_neighs = self._set_neighs_general_array
            elif self.level == 0:
                self.set_neighs = self._set_neighs_number
            elif self.level == 1:
                self.set_neighs = self._set_neighs_array_lvl1
            elif self.level == 2:
                self.set_neighs = self._set_neighs_array_lvl2
            elif self.level == 3:
                self.set_neighs = self._set_neighs_array_lvl3
        elif type_neighs == 'list':
            # Format get neighs
            if self._constant_neighs:
                if self.staticneighs:
                    self.get_neighs = self._get_neighs_array_static
                else:
                    self.get_neighs = self._get_neighs_array_dynamic
            else:
                if self.staticneighs:
                    self.get_neighs = self._get_neighs_list_static
                else:
                    self.get_neighs = self._get_neighs_list_dynamic
            # Format set neighs
            if self.level is None:
                self.set_neighs = self._set_neighs_general_list
            elif self.level == 0:
                self.set_neighs = self._set_neighs_number
            elif self.level == 1:
                self.set_neighs = self._set_neighs_list_only
            elif self.level == 2:
                self.set_neighs = self._set_neighs_list_list
            elif self.level == 3:
                self.set_neighs = self._set_neighs_list_list_list
        elif type_neighs == 'slice':
            self.set_neighs = self._set_neighs_slice
            self.get_neighs = self._get_neighs_slice
            self.staticneighs_set = True

    def set_structure(self, format_structure=None):
        """Set the structure in which the neighbourhood information will be
        given.

        Parameters
        ----------
        format_structure: str, optional (default=None)
            the type of structure in which we are going to set the
            neighbourhood information.

        """
        if format_structure is None:
            self._set_info = self._set_general
        elif format_structure == 'raw':
            self._set_info = self._set_raw_structure
            self.ifdistance = False
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        elif format_structure == 'tuple':
            self._set_info = self._set_tuple_structure
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        elif format_structure == 'tuple_only':
            self._set_info = self._set_tuple_only_structure
        elif format_structure == 'tuple_k':
            self._set_info = self._set_tuple_k_structure
        elif format_structure == 'tuple_tuple':
            self._set_info = self._set_tuple_tuple_structure
        elif format_structure == 'list_tuple_only':
#            assert(self.level == 2)
            self._set_info = self._set_list_tuple_only_structure
            self.staticneighs_set = False
            if self.level != 2:
                raise Exception("Not correct inputs.")
            else:
                self.level = 3
        elif format_structure == 'tuple_list_tuple':
#            assert(self.level == 2)
            self._set_info = self._set_tuple_list_tuple_structure
            self.staticneighs_set = False
            if self.level != 2:
                raise Exception("Not correct inputs.")
            else:
                self.level = 3

    ###########################################################################
    ################################# FORMATS #################################
    ###########################################################################

    ############################### Formatters ################################
    ###########################################################################
    def _format_globalpars(self, staticneighs, ifdistance, format_level):
        """Global information non-mutable and mutable in order to force or keep
        other information and functions.

        Parameters
        ----------
        staticneighs: boolean
            if there is constant neighbourhood across the perturbations.
        ifdistance: boolean
            if we set the distance or the relative position information.
        format_level: int
            the level in which the information of the neighborhood will be set.

        """
        ## Basic information how it will be input neighs_info
        self.level = format_level
        ## Global known information about relative position
        self.ifdistance = ifdistance
        ## Global known information about get information
        self.staticneighs = staticneighs
        ## Setting changable information about static neighs setting
        self.staticneighs_set = None
        if self.level is None:
            self.staticneighs_set = None
        elif self.level <= 2:
            self.staticneighs_set = True
        if self.level == 3:
            self.staticneighs_set = False

    def _format_setters(self, format_structure, type_neighs=None,
                        type_sp_rel_pos=None, format_set_iss=None):
        """Format the setter functions.

        Parameters
        ----------
        format_structure: str, optional
            the type of structure in which we are going to set the
            neighbourhood information.
        type_neighs: str optional (default=None)
            the type of object describing the neighs of the neighbourhood.
        type_sp_rel_pos: str optional (default=None)
            the type of object describing the relative position of the
            neighbourhood.
        format_set_iss: str optional (default=None)
            in which format of elements iss we set.

        """
        ## 1. Format structure
        self.set_structure(format_structure)
        ## 2. Set types
        self.set_types(type_neighs, type_sp_rel_pos)
        ## 3. Post-format
        if self._constant_neighs:
            self._main_postformat = self._cte_postformat
        else:
            self._main_postformat = self._null_postformat
        self._iss_postformat = self._assert_iss_postformat
        self._ks_postformat = self._assert_ks_postformat
        if self._constant_neighs and type_neighs != 'slice':
            self._idxs_postformat = self._idxs_postformat_array
        else:
            self._idxs_postformat = self._idxs_postformat_null

        ## 4. Format iss
        self._format_set_iss(format_set_iss)

        ## 5. Format set ks
        if self.staticneighs:
            self.set_ks = self._set_ks_static
        else:
            self.set_ks = self._set_ks_dynamic

        ## 6. General set
        self.set = self._general_set

    def _format_set_iss(self, format_set_iss=None):
        """Format the setter iss function.

        Parameters
        ----------
        format_set_iss: str optional (default=None)
            in which format of elements iss we set.

        """
        ## Format iss
        if format_set_iss is None or format_set_iss == 'general':
            self._set_iss = self._general_set_iss
        elif format_set_iss == 'null':
            self._set_iss = self._null_set_iss
        elif format_set_iss == 'int':
            self._set_iss = self._int_set_iss
        elif format_set_iss == 'list':
            self._set_iss = self._list_set_iss

    def _format_getters(self, format_get_info=None, format_get_k_info=None):
        """Function to program this class according to the stored idxs.

        Parameters
        ----------
        format_get_info: str optional (default=None)
            in which format the information is returned to the user.
        format_get_k_info: str optional (default=None)
            in which format of the ks we set.

        """
        ## Get info setting
        if format_get_k_info is None:
            self.get_k = self._general_get_k
        elif format_get_k_info == "default":
            self.get_k = self._default_get_k
        elif format_get_k_info == "general":
            self.get_k = self._general_get_k
        elif format_get_k_info == "list":
            self.get_k = self._list_get_k
        elif format_get_k_info == "integer":
            self.get_k = self._integer_get_k
        ## Get information setting
        if format_get_info is None:
            self.get_information = self._general_get_information
        elif format_get_info == "default":
            self.get_information = self._default_get_information
        elif format_get_info == "general":
            self.get_information = self._general_get_information
        ## Other getters
        if self.staticneighs:
            self.get_copy_iss = self._staticneighs_get_copy_iss
            self.get_copy_iss_by_ind = self._staticneighs_get_copy_iss_by_ind
        else:
            self.get_copy_iss = self._notstaticneighs_get_copy_iss
            self.get_copy_iss_by_ind =\
                self._notstaticneighs_get_copy_iss_by_ind

    def _postformat(self):
        """Format properly."""
        self._main_postformat()
        self._iss_postformat()
        self._assert_ks_postformat()
        self._idxs_postformat()

    def _cte_postformat(self):
        """To array because of constant neighs."""
#        if type(self.idxs) == list:
#            self.idxs = np.array(self.idxs)
        if self.sp_relative_pos is not None:
            if type(self.sp_relative_pos) == list:
                self.sp_relative_pos = np.array(self.sp_relative_pos)

    def _assert_iss_postformat(self):
        """Assert if the iss is correctly formatted, if not, format properly.
        """
        if type(self.idxs) in [list, np.ndarray]:
#            print self.idxs, self.iss, self.set_neighs
            if self.staticneighs:
                ### WARNING: Redefinition of iss.
                if len(self.idxs) != len(self.iss):
                    if len(self.idxs[0]) == len(self.iss):
                        self.idxs = self.idxs[0]
                    else:
                        self.iss = range(len(self.idxs))
            else:
                assert(all([len(k) == len(self.idxs[0]) for k in self.idxs]))

    def _assert_ks_postformat(self):
        """Assert proper postformatting for the ks."""
        if type(self.idxs) in [list, np.ndarray]:
            if self.ks is None:
                if self.staticneighs:
                    pass
                else:
                    self.ks = range(len(self.idxs))
            if self.staticneighs:
                pass
            else:
#                print self.ks, self.idxs, self.set_neighs, self.set_sp_rel_pos
                assert(len(self.ks) == len(self.idxs))
        ## Defining functions
        if self.sp_relative_pos is not None and self.staticneighs:
            self.get_sp_rel_pos = self._static_get_rel_pos
        elif not self.staticneighs:
            if type(self.sp_relative_pos) == list:
                self.get_sp_rel_pos = self._dynamic_rel_pos_list
            else:
                self.get_sp_rel_pos = self._dynamic_rel_pos_array
        if self.sp_relative_pos is None:
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        ## Ensure correct k_ret
        if np.max(self.ks) > self._kret:
            self._kret = np.max(self.ks)

#    def _array_ele_postformat(self, ele):
#        return np.array(ele)
#
#    def _null_ele_postformat(self, ele):
#        return ele

    def _null_postformat(self):
        """Not change anything."""
        pass

    def _idxs_postformat_array(self):
        """The neighs information postformatting. It format in an array-form
        the neighs stored in the instance.
        """
        self.idxs = np.array(self.idxs)

    def _idxs_postformat_null(self):
        """The neighs information postformatting. It doesnt change the format.
        """
        pass

    ###########################################################################
    ################################## SETS ###################################
    ###########################################################################

    ########################### Setters candidates ############################
    ###########################################################################
    def _general_set(self, neighs_info, iss=None):
        """General set.

        Parameters
        ----------
        neighs_info: int, float, slice, np.ndarray, list, tuple or instance
            the neighbourhood information given with the proper indicated
            structure.
        iss: list or np.ndarray (default=None)
            the indices of the elements we stored their neighbourhood.

        """
        ## Set function
        self._preset(neighs_info, iss)
        ## Post-set functions
        self._postset()
        self.assert_goodness()

    def _preset(self, neighs_info, iss=None):
        """Set the class.

        Parameters
        ----------
        neighs_info: int, float, slice, np.ndarray, list, tuple or instance
            the neighbourhood information given with the proper indicated
            structure.
        iss: list or np.ndarray (default=None)
            the indices of the elements we stored their neighbourhood.

        """
        self._reset_stored()
        self._set_iss(iss)
        self._set_info(neighs_info)
        self._postformat()

    def _postset(self):
        """Postsetting class."""
        if type(self.idxs) == np.ndarray:
            pass
        if type(self.idxs) == slice:
            self.get_neighs = self._get_neighs_slice
        elif type(self.idxs) == np.ndarray:
#            if len(self.idxs.shape) == 3 and self.ks is None:
#                self.ks = list(range(len(self.idxs)))
#            else:
#                self.staticneighs_set = True
            if self.staticneighs:
                self.get_neighs = self._get_neighs_array_static
            else:
                self.get_neighs = self._get_neighs_array_dynamic
        elif type(self.idxs) == list:
            if self.staticneighs:
                self.get_neighs = self._get_neighs_list_static
            else:
                self.get_neighs = self._get_neighs_list_dynamic
        ## Format coreget by iss
        if type(self.idxs) == slice:
            self._staticneighs_get_corestored_by_inds =\
                self._staticneighs_get_corestored_by_inds_slice
            self._notstaticneighs_get_corestored_by_inds =\
                self._notstaticneighs_get_corestored_by_inds_slice
        else:
            self._staticneighs_get_corestored_by_inds =\
                self._staticneighs_get_corestored_by_inds_notslice
            self._notstaticneighs_get_corestored_by_inds =\
                self._notstaticneighs_get_corestored_by_inds_notslice

    def _set_init(self):
        """Reset variables to default."""
        ## Main information
        self.idxs = None
        self.sp_relative_pos = None
        ## Auxiliar information
        self.ks = None
        self.iss = [0]
        ## Class structural information
        self._setted = False
        self._constant_rel_pos = False
        self.staticneighs = None
        self.staticneighs_set = None

    def _reset_stored(self):
        """Reset the stored parameters and neighbourhood information."""
        ## Main information
        self.idxs = None
        self.sp_relative_pos = None
        self._setted = False
        self.ks = None
        self.iss = [0]

    def _set_general(self, neighs_info):
        """Setting neighs info with heterogenous ways to do it.

        Parameters
        ----------
        neighs_info: int, float, slice, np.ndarray, list, tuple or instance
            the neighbourhood information given with the proper indicated
            structure. The standards of the inputs are:
                * neighs [int, float, list, slice or np.ndarray]
                * (i, k)
                * (neighs, k)
                * (neighs_info, k) where neighs_info is a tuple which could
                contain (neighs, dists) or (neighs,)
                * neighs_info in the form of pst.Neighs_Info
        """
        ## 0. Format inputs
        # If int is a neighs
        if type(neighs_info) in [int, float, np.int32, np.int64, np.float]:
            self._set_neighs_number(neighs_info)
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        # If slice is a neighs
        elif type(neighs_info) == slice:
            self._set_neighs_slice(neighs_info)
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        # If array is a neighs
        elif type(neighs_info) == np.ndarray:
            self._set_neighs_general_array(neighs_info)
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        # If int could be neighs or list of tuples
        elif type(neighs_info) == list:
            self._set_structure_list(neighs_info)
        # If tuple there are more information than neighs
        elif type(neighs_info) == tuple:
            self._set_structure_tuple(neighs_info)
        else:
            assert(type(neighs_info).__name__ == 'instance')
            ## Substitution main information
            self.idxs = neighs_info.idxs
            self.ks = neighs_info.ks
            self.iss = neighs_info.iss
            ## Copying class information
            self._constant_neighs = neighs_info._constant_neighs
            self._kret = neighs_info._kret
            self._n = neighs_info._n
            self.format_set_info = neighs_info.format_set_info
            self.format_get_info = neighs_info.format_get_info
            self._format_globalpars(neighs_info.staticneighs,
                                    neighs_info.ifdistance, neighs_info.level)
            self._format_setters(*neighs_info.format_set_info)
            self._format_getters(*neighs_info.format_get_info)
            self._format_joining_functions()

    ############################## Set Structure ##############################
    ###########################################################################
    def _set_raw_structure(self, key):
        """Set the neighbourhood information in a form of raw structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * neighs{any form}

        """
        self.set_neighs(key)
        self.ifdistance = False

    def _set_structure_tuple(self, key):
        """Set the neighbourhood information in a form of tuple general.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * (neighs, )
                * (neighs_info{any form}, ks)
                * (neighs_info{list of typle only}, ks)
                * (neighs{any form}, sp_relative_pos{any form})
                * ((neighs{any form}, sp_relative_pos{any form}), ks)
                * (neighs_info{list of typle only}, ks)

        """
        if len(key) == 2:
            msg = "Ambiguous input in `set` function of pst.Neighs_Info."
            warnings.warn(msg, SyntaxWarning)
            if type(key[0]) == tuple:
                self.ks = list(np.array([key[1]]).ravel())
                self._set_structure_tuple(key[0])
            else:
                aux_bool = type(key[0]) in [np.ndarray, list]
                if type(key[0]) == list and type(key[0][0]) == tuple:
                    self._set_tuple_list_tuple_structure(key)
                elif type(key[0]) == type(key[1]) and aux_bool:
                    if len(key[0]) == len(key[1]):
                        self._set_tuple_only_structure(key)
                    else:
                        self.ks = list(np.array(key[1]))
                        self.set_neighs(key[0])
                else:
                    self._set_tuple_only_structure(key)
        else:
            self.set_neighs(key[0])

    def _set_tuple_structure(self, key):
        """Set the neighbourhood information in a form of tuple structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * (neighs_info{any form}, ks)

        """
        if len(key) == 2:
            self.ks = list(np.array(key[1]))
        self.set_neighs(key[0])

    def _set_tuple_only_structure(self, key):
        """Set the neighbourhood information in a form of tuple only structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * (neighs{any form}, sp_relative_pos{any form})

        """
        self.set_neighs(key[0])
        if len(key) == 2:
            self.set_sp_rel_pos(key[1])
        elif len(key) > 2:
            raise TypeError("Not correct input.")

    def _set_tuple_tuple_structure(self, key):
        """Set the neighbourhood information in a form of tuple tuple
        structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * ((neighs{any form}, sp_relative_pos{any form}), ks)

        """
        if len(key) == 2:
            ks = [key[1]] if type(key[1]) == int else key[1]
            self.ks = list(np.array([ks]).ravel())
        self._set_tuple_only_structure(key[0])

#    def _set_tuple_list_tuple_only(self, key):
#        """
#        * (neighs_info{list of typle only}, ks)
#        """
#        self.ks = list(np.array(key[1]))
#        self._set_list_tuple_only_structure(key[0])

    def _set_tuple_k_structure(self, key):
        """Set the neighbourhood information in a form of tuple structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * (idxs, ks)

        """
        self.ks = [key[1]] if type(key[1]) == int else key[1]
        self.set_neighs(key[0])

    def _set_structure_list(self, key):
        """Set the neighbourhood information in a form of general list
        structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * [neighs_info{tuple form}]

        """
        if len(key) == 0:
            self.set_neighs = self._set_neighs_general_list
            self.set_neighs(key)
        elif type(key[0]) == tuple:
            self._set_info = self._set_list_tuple_only_structure
            self._set_info(key)
        elif type(key[0]) == list:
            if self._constant_neighs:
                if self.staticneighs:
                    self.get_neighs = self._get_neighs_array_static
                else:
                    self.get_neighs = self._get_neighs_array_dynamic
            else:
                if self.staticneighs:
                    self.get_neighs = self._get_neighs_list_static
                else:
                    self.get_neighs = self._get_neighs_list_dynamic
            # Format set neighs
            self.set_neighs = self._set_neighs_general_list
            self.set_neighs(key)
        elif type(key[0]) == np.ndarray:
            self.set_neighs = self._general_set_neighs
            self.set_neighs(np.array(key))
        elif type(key[0]) in [int, float, np.int32, np.int64]:
            self.level = 1
            self._set_info = self._set_raw_structure
            self.ifdistance = False
            self.set_sp_rel_pos = self._null_set_rel_pos
            if self.staticneighs:
                self.get_neighs = self._get_neighs_array_static
            else:
                self.get_neighs = self._get_neighs_array_dynamic
            # Format set neighs
            self.set_neighs = self._set_neighs_array_lvl1
            self.set_neighs(np.array(key))

    def _set_list_tuple_only_structure(self, key):
        """Set the neighbourhood information in a form of list tuple only
        structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * [(neighs{any form}, sp_relative_pos{any form})]

        """
        ## Change to list and whatever it was
        self.set_neighs([e[0] for e in key])
        self.set_sp_rel_pos([e[1] for e in key])

    def _set_tuple_list_tuple_structure(self, key):
        """Set the neighbourhood information in a form of tuple, list tuple
        structure.

        Parameters
        ----------
        neighs_info: tuple
            the neighborhood information for each element `i` and perturbations
            `k`. The standards to set that information are:
                * (neighs_info{list of typle only}, ks)

        """
        self.ks = [key[1]] if type(key[1]) == int else key[1]
        if not self.staticneighs:
            assert(len(key[0]) == len(self.ks))
        self._set_list_tuple_only_structure(key[0])

    ############################### Set Neighs ################################
    ###########################################################################
    ## After that has to be set:
    # - self.idxs
    # - self.ks
    #
    def _general_set_neighs(self, key):
        """General setting of only neighs.

        Parameters
        ----------
        neighs: list or np.ndarray
            the neighs information for each element `i`. The standards to set
            that information are:
                * neighs {number form}
                * neighs {list form}
                * neighs {array form}

        """
        if type(key) == list:
            self._set_neighs_general_list(key)
        elif type(key) == np.ndarray:
            self._set_neighs_general_array(key)
        elif type(key) in inttypes:
            self._set_neighs_number(key)
        else:
#            print key
            raise TypeError("Incorrect neighs input in pst.Neighs_Info")

    def _set_neighs_number(self, key):
        """Only one neighbor expressed in a number way.

        Parameters
        ----------
        neighs: int
            the neighborhood information for each element `i`. The standards to
            set that information are:
                * indice{int form}

        """
        if self.staticneighs:
            self.idxs = np.array([[key]]*len(self.iss))
        else:
            if self.ks is None:
                self.ks = range(1)
            len_ks = len(self.ks)
            self.idxs = np.array([[[key]]*len(self.iss)]*len_ks)
        self._constant_neighs = True
        self._setted = True

    def _set_neighs_slice(self, key):
        """Set neighs in a slice-form.

        Parameters
        ----------
        neighs: slice
            the neighs information for each element `i`. The standards to set
            that information are:
                * indices{slice form}

        """
        ## Condition to use slice type
        self._constant_neighs = True
        self.ks = range(1) if self.ks is None else self.ks
        ## Possible options
        if key is None:
            self.idxs = slice(0, self._n, 1)
        elif isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = self._n if key.stop is None else key.stop
            stop = self._n if key.stop > 10*16 else key.stop
            step = 1 if key.step is None else key.step
            self.idxs = slice(start, stop, step)
        elif type(key) in inttypes:
            self.idxs = slice(0, key, 1)
        elif type(key) == tuple:
            self.idxs = slice(key[0], key[1], 1)
        self._setted = True

    def _set_neighs_array_lvl1(self, key):
        """Set neighs as a array level 1 form.

        Parameters
        ----------
        neighs: np.ndarray
            the neighs information for each element `i`. The standards to set
            that information are:
                * indices{np.ndarray form} shape: (neighs)

        """
        #sh = key.shape
        ## If only array of neighs
        if self.staticneighs:
            self.idxs = np.array([key for i in range(len(self.iss))])
        else:
            self.ks = range(1) if self.ks is None else self.ks
            len_ks = len(self.ks)
            self.idxs = np.array([[key for i in range(len(self.iss))]
                                  for i in range(len_ks)])
        self._setted = True

    def _set_neighs_array_lvl2(self, key):
        """Set neighs as array level 2 form.

        Parameters
        ----------
        neighs: np.ndarray
            the neighs information for each element `i`. The standards to set
            that information are:
                * indices{np.ndarray form} shape: (iss, neighs)

        """
        sh = key.shape
        ## If only iss and neighs
        self.idxs = key
        if self.staticneighs:
            self.idxs = np.array(key)
        else:
            len_ks = len(self.ks) if self.ks is not None else 1
            self.ks = range(1) if self.ks is None else self.ks
            self.idxs = np.array([key for k in range(len_ks)])
        self._setted = True
        if sh[0] != len(self.iss):
            self.iss = list(range(sh[0]))

    def _set_neighs_array_lvl3(self, key):
        """Set neighs as array level 3 form.

        Parameters
        ----------
        neighs: np.ndarray
            the neighs information for each element `i`. The standards to set
            that information are:
                * indices{np.ndarray form} shape: (ks, iss, neighs)

        """
        self.idxs = np.array(key)
        self.ks = range(len(self.idxs)) if self.ks is None else self.ks
        if self.staticneighs:
            self.idxs = np.array(key[0])
            if len(self.idxs) != len(self.iss):
                self.iss = list(range(len(self.idxs)))
        else:
            if len(self.idxs[0]) != len(self.iss):
                self.iss = list(range(len(self.idxs[0])))
        self._setted = True

    def _set_neighs_general_array(self, key):
        """Set neighs as a general array form.

        Parameters
        ----------
        neighs: np.ndarray
            the neighs information for each element `i`. The standards to set
            that information are:
                * indices{np.ndarray form} shape: (neighs)
                * indices{np.ndarray form} shape: (iss, neighs)
                * indices{np.ndarray form} shape: (ks, iss, neighs)

        """
        key = np.array([key]) if type(key) in inttypes else key
        sh = key.shape
        ## If only array of neighs
        if len(sh) == 0:
            self._set_neighs_number(key)
#            self._setted = False
#            if self.staticneighs:
#                self.idxs = np.array([[]])
#            else:
#                self.idxs = np.array([[[]]])
        elif len(sh) == 1:
            self._set_neighs_array_lvl1(key)
        ## If only iss and neighs
        elif len(sh) == 2:
            self._set_neighs_array_lvl2(key)
        elif len(sh) == 3:
            self._set_neighs_array_lvl3(key)

    def _set_neighs_general_list(self, key):
        """Set neighs as a general list form.

        Parameters
        ----------
        neighs: list
            the neighs information for each element `i`. The standards to set
            that information are:
                * indices {list of list form [neighs]} [neighs]
                * [neighs_info{array-like form}, ...] [iss][neighs]
                * [neighs_info{array-like form}, ...] [ks][iss][neighs]

        """
        ### WARNING: NOT WORK WITH EMPTY NEIGHS
        if '__len__' not in dir(key):
            self._set_neighs_number(key)
        else:
            if len(key) == 0:
                self._set_neighs_list_only(key)
            elif '__len__' not in dir(key[0]):
                self._set_neighs_list_only(key)
            else:
                if all([len(key[i]) == 0 for i in range(len(key))]):
                    self._setted = False
                    if self.staticneighs:
                        self.idxs = np.array([[]])
                    else:
                        self.idxs = np.array([[[]]])
                elif '__len__' not in dir(key[0][0]):
                    self._set_neighs_list_list(key)
                else:
                    self._set_neighs_list_list_list(key)

    def _set_neighs_list_only(self, key):
        """Set the level 1 list

        Parameters
        ----------
        neighs: list
            the neighs information for each element `i`. The standards to set
            that information are:
                * indices {list of list form [neighs]} [neighs]

        """
        self._set_neighs_array_lvl1(np.array(key))

    def _set_neighs_list_list(self, key):
        """Set the level 2 list.

        Parameters
        ----------
        neighs: list
            the neighs information for each element `i`. The standards to set
            that information are:
                * [neighs_info{array-like form}, ...] [iss][neighs]

        """
        if self._constant_neighs:
            key = np.array(key)
        if self.staticneighs:
            self.idxs = key
            self.ks = range(1) if self.ks is None else self.ks
        else:
            self.ks = range(1) if self.ks is None else self.ks
            len_ks = len(self.ks)
            self.idxs = [key for k in range(len_ks)]
            if type(key) == np.ndarray:
                self.idxs = np.array(self.idxs)
        if len(self.iss) != len(key):
            if len(self.iss) != len(key):
                self.iss = range(len(key))
#        if len(self.idxs[0]) > 0:
#            self.iss = list(range(len(self.idxs)))
        self._setted = True

    def _set_neighs_list_list_list(self, key):
        """Set neighs as a level 3 list form.

        Parameters
        ----------
        neighs: list
            the neighs information for each element `i`. The standards to set
            that information are:
                * [neighs_info{array-like form}, ...] [ks][iss][neighs]

        """
        self.ks = list(range(len(key))) if self.ks is None else self.ks
        if self._constant_neighs:
            self.idxs = np.array(key)
        else:
            self.idxs = key
        if len(self.idxs[0]) != len(self.iss):
            self.iss = list(range(len(self.idxs[0])))
        if self.staticneighs:
            self.idxs = self.idxs[0]
        self._setted = True

    ########################### Set Sp_relative_pos ###########################
    ###########################################################################
    def _general_set_rel_pos(self, rel_pos):
        """Set the general relative position.

        Parameters
        ----------
        rel_pos: int, float, list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * None
                * list of arrays len(iss) -> unique rel_pos for ks
                * list of lists of arrays -> complete

        """
        if rel_pos is None or self.ifdistance is False:
            self._null_set_rel_pos(rel_pos)
            self.get_sp_rel_pos = self._null_get_rel_pos
        elif type(rel_pos) == list:
            self._set_rel_pos_general_list(rel_pos)
        elif type(rel_pos) == np.ndarray:
            self._set_rel_pos_general_array(rel_pos)
        elif type(rel_pos) in [float, int, np.float, np.int32, np.int64]:
            self._set_rel_pos_number(rel_pos)
        else:
#            print rel_pos
            msg = "Incorrect relative position input in pst.Neighs_Info"
            raise TypeError(msg)

    def _set_rel_pos_general_list(self, rel_pos):
        """Set of relative position in a general list form.

        Parameters
        ----------
        rel_pos: list
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * None
                * list of arrays len(iss) -> unique rel_pos for ks
                * list of lists of arrays -> complete

        """
        if self.level is not None:
            if self.level == 0:
                self._set_rel_pos_dim(rel_pos)
            elif self.level == 1:
                self._list_only_set_rel_pos(rel_pos)
            elif self.level == 2:
                self._list_list_only_set_rel_pos(rel_pos)
            elif self.level == 3:
                self._list_list_set_rel_pos(rel_pos)
        else:
            if len(rel_pos) == 0:
                self._set_rel_pos_number(rel_pos)
            elif type(rel_pos[0]) not in [list, np.ndarray]:
                self._list_only_set_rel_pos(rel_pos)
            else:
                if len(rel_pos[0]) == 0:
                    self._list_only_set_rel_pos(rel_pos)
                elif type(rel_pos[0][0]) not in [list, np.ndarray]:
                    self._list_only_set_rel_pos(rel_pos)
                else:
                    if len(rel_pos[0][0]) == 0:
                        self._list_list_only_set_rel_pos(rel_pos)
                    elif type(rel_pos[0][0][0]) not in [list, np.ndarray]:
                        self._list_list_only_set_rel_pos(rel_pos)
                    else:
                        self._list_list_set_rel_pos(rel_pos)

    def _null_set_rel_pos(self, rel_pos):
        """Not consider the input.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.

        """
        self.get_sp_rel_pos = self._null_get_rel_pos

    def _set_rel_pos_number(self, rel_pos):
        """Number set pos.

        Parameters
        ----------
        rel_pos: int or float
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * int or float

        """
        self.sp_relative_pos = self._set_rel_pos_dim([rel_pos])

    def _set_rel_pos_dim(self, rel_pos):
        """Set rel pos with zero level.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * rel_pos{array or list form} [dim]

        """
        if not '__len__' in dir(rel_pos):
            rel_pos = np.array([rel_pos])
        if self.staticneighs:
            rel_pos_f = []
            for i in range(len(self.idxs)):
                rel_pos_i = [rel_pos for nei in range(len(self.idxs[i]))]
                rel_pos_f.append(rel_pos_i)
        else:
            rel_pos_f = []
            for k in range(len(self.idxs)):
                rel_pos_k = []
                for i in range(len(self.idxs[k])):
                    n_nei = len(self.idxs[k][i])
                    rel_pos_k.append([rel_pos for nei in range(n_nei)])
                rel_pos_f.append(rel_pos_k)

        if self._constant_neighs:
            rel_pos_f = np.array(rel_pos_f)
        self.sp_relative_pos = rel_pos_f
#        self.sp_relative_pos = np.array([[[rel_pos]]])
#        self.get_sp_rel_pos = self._constant_get_rel_pos
#        self.staticneighs = True

    def _set_rel_pos_general_array(self, rel_pos):
        """Array set rel pos.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * rel_pos{np.ndarray form} shape: (neighs, dim)
                * rel_pos{np.ndarray form} shape: (iss, neighs, dim)
                * rel_pos{np.ndarray form} shape: (ks, iss, neighs, dim)

        """
        n_shape = len(rel_pos.shape)
        if n_shape == 2:
            self._array_only_set_rel_pos(rel_pos)
        elif n_shape == 3:
            self._array_array_set_rel_pos(rel_pos)
        elif n_shape == 4:
            self._array_array_array_set_rel_pos(rel_pos)

    def _array_only_set_rel_pos(self, rel_pos):
        """Set the array form relative position.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * Array only. [nei][dim] or [nei]

        """
        ## Preformatting
        rel_pos = np.array(rel_pos)
        if len(rel_pos.shape) == 1:
            rel_pos = rel_pos.reshape((len(rel_pos), 1))
        n_iss = len(self.iss)
        sp_relative_pos = np.array([rel_pos for i in range(n_iss)])
        ## Not staticneighs
        if not self.staticneighs:
            n_k = len(self.idxs)
            sp_relative_pos = np.array([sp_relative_pos for i in range(n_k)])
        self.sp_relative_pos = sp_relative_pos

    def _array_array_set_rel_pos(self, rel_pos):
        """Set the array-array (level 2) relative position.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                *Array or arrays. [iss][nei][dim] or [nei].

        """
#        self.staticneighs = True
        if self.staticneighs:
            self.sp_relative_pos = np.array(rel_pos)
        else:
            len_ks = 1 if self.ks is None else len(self.ks)
            self.sp_relative_pos = np.array([rel_pos for k in range(len_ks)])

    def _array_array_array_set_rel_pos(self, rel_pos):
        """Set the level 3 array relative position.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * Array or arrays. [ks][iss][nei][dim] or [ks][nei].

        """
        if self.staticneighs:
            self.sp_relative_pos = rel_pos[0]
        else:
            self.sp_relative_pos = rel_pos

    def _list_only_set_rel_pos(self, rel_pos):
        """List only relative pos. Every iss and ks has the same neighs with
        the same relative information.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * [nei][dim] or [nei]

        """
        self._array_only_set_rel_pos(rel_pos)

    def _list_list_only_set_rel_pos(self, rel_pos):
        """List list only relative pos. Every ks has the same neighs with the
        same relative information.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                *[iss][nei][dim] or [iss][nei]

        """
        if self.staticneighs is not True:
            assert(self.ks is not None)
            n_ks = len(self.ks)
            self.sp_relative_pos = [rel_pos]*n_ks
        else:
            self.sp_relative_pos = rel_pos

    def _list_list_set_rel_pos(self, rel_pos):
        """List list list relative pos.

        Parameters
        ----------
        rel_pos: list or np.ndarray
            the relative position of the neighbourhood respect the centroid.
            The standard inputs form are:
                * [ks][iss][nei][dim] or [ks][iss][nei]

        """
        if self.staticneighs:
            self.sp_relative_pos = rel_pos[0]
        else:
            self.sp_relative_pos = rel_pos

    ############################### Setter iss ################################
    ###########################################################################
    def _general_set_iss(self, iss):
        """General set iss input.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        if type(iss) == int:
            self._int_set_iss(iss)
        elif type(iss) in [list, np.ndarray]:
            self._list_set_iss(iss)
        else:
            if type(self.idxs) in [list, np.ndarray]:
                if self.staticneighs:
                    if len(self.iss) != len(self.idxs):
                        self.iss = range(len(self.idxs))
                else:
                    if len(self.iss) != len(self.idxs[0]):
                        self.iss = range(len(self.idxs[0]))

    def _int_set_iss(self, iss):
        """Input iss always integer.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        self.iss = [iss]

    def _list_set_iss(self, iss):
        """Input iss always array-like.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        self.iss = list(iss)

    def _null_set_iss(self, iss):
        """Not consider the input.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        pass

    ###########################################################################
    ################################## GETS ###################################
    ###########################################################################

    ############################# Getter rel_pos ##############################
    ###########################################################################
    def _general_get_rel_pos(self, k_is=[0]):
        """Get the relative position.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        if self.sp_relative_pos is None:
            return self._null_get_rel_pos(k_is)
        elif self.staticneighs:
            return self._static_get_rel_pos(k_is)
#        elif self._constant_rel_pos:
#            return self._constant_get_rel_pos(k_is)
        else:
            if type(self.sp_relative_pos) == list:
                return self._dynamic_rel_pos_list(k_is)
            else:
                return self._dynamic_rel_pos_array(k_is)

    def _null_get_rel_pos(self, k_is=[0]):
        """Get the relative position.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        return [[None]*len(self.iss)]*len(k_is)

#    def _constant_get_rel_pos(self, k_is=[0]):
#        neighs = self.get_neighs(k_is)
#        rel_pos = []
#        for k in range(len(neighs)):
#            rel_pos_k = []
#            for i in range(len(neighs[k])):
#                rel_pos_k.append(len(neighs[k][i])*[self.sp_relative_pos])
#            rel_pos.append(rel_pos_k)
#        if self._constant_neighs:
#            rel_pos = np.array(rel_pos)
#        return rel_pos

    def _static_get_rel_pos(self, k_is=[0]):
        """Get the relative position.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        return [self.sp_relative_pos for k in k_is]

#    def _static_rel_pos_list(self, k_is=[0]):
#        return self.sp_relative_pos*len(k_is)
#
#    def _static_rel_pos_array(self, k_is=[0]):
#        return np.array([self.sp_relative_pos for i in range(len(k_is))])

    def _dynamic_rel_pos_list(self, k_is=[0]):
        """Get the relative position.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
#        [[e[k_i] for e in self.sp_relative_pos] for k_i in k_is]
        return [self.sp_relative_pos[i] for i in k_is]

    def _dynamic_rel_pos_array(self, k_is=[0]):
        """Get the relative position.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
#        [[e[k_i] for e in self.sp_relative_pos] for k_i in k_is]
        return [self.sp_relative_pos[i] for i in k_is]

    ################################ Getters k ################################
    ###########################################################################
    def _general_get_k(self, k=None):
        """General get k.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated unformatted.

        Returns
        -------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated formatted.

        """
        ## Format k
        if k is None:
            ks = self._default_get_k()
        elif type(k) in [np.ndarray, list]:
            ks = self._list_get_k(k)
        elif type(k) in inttypes:
            ks = self._integer_get_k(k)
        return ks

    def _default_get_k(self, k=None):
        """Default get ks.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated unformatted.

        Returns
        -------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated formatted.

        """
        if self.ks is None:
            return [0]
        else:
            return self.ks

    def _integer_get_k(self, k):
        """Integer get k.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated unformatted.

        Returns
        -------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated formatted.

        """
        if type(k) == list:
            return [self._integer_get_k(e)[0] for e in k]
        if k >= 0 and k <= self._kret:
            ks = [k]
        else:
            raise TypeError("k index out of bounds.")
        return ks

    def _list_get_k(self, k):
        """List get k.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated unformatted.

        Returns
        -------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated formatted.

        """
        ks = [self._integer_get_k(k_i)[0] for k_i in k]
        return ks

    def _get_k_indices(self, ks):
        """List of indices of ks.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray
            the perturbations indices associated with the returned information.

        Returns
        -------
        idx_ks: list
            the associated indices to the perturbation indices. Get the index
            order.

        """
        if self.staticneighs:
            idx_ks = ks
        else:
            idx_ks = [self.ks.index(e) for e in ks]
        return idx_ks

    ############################ Getters information ##########################
    ###########################################################################
    def _general_get_information(self, k=None):
        """Get information stored in this class.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=None)
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.
        ks: int, slice, list or np.ndarray (default=None)
            the perturbations indices associated with the returned information.
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        ## Format k
        ks = self.get_k(k)
        idx_ks = self._get_k_indices(ks)
        ## Get iss
        iss = self.iss
        ## Format idxs
        assert(type(idx_ks) == list)
        neighs = self.get_neighs(idx_ks)
        sp_relative_pos = self.get_sp_rel_pos(idx_ks)
        self.check_output_standards(neighs, sp_relative_pos, ks, iss)
#        print '3'*50, neighs, sp_relative_pos, ks, iss
        return neighs, sp_relative_pos, ks, iss

    def _default_get_information(self, k=None):
        """For the unset instances.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=None)
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.
        ks: int, slice, list or np.ndarray (default=None)
            the perturbations indices associated with the returned information.
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        raise Exception("Information not set in pst.Neighs_Info.")

    ################################ Get neighs ###############################
    def _get_neighs_general(self, k_is=[0]):
        """General getting neighs.

        Parameters
        ----------
        ks: int, slice, list or np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.

        """
        if type(self.idxs) == slice:
            neighs = self._get_neighs_slice(k_is)
        elif type(self.idxs) == np.ndarray:
            if self.staticneighs:
                neighs = self._get_neighs_array_static(k_is)
            else:
                neighs = self._get_neighs_array_dynamic(k_is)
        elif type(self.idxs) == list:
            if self.staticneighs:
                neighs = self._get_neighs_list_static(k_is)
            else:
                neighs = self._get_neighs_list_dynamic(k_is)
#        else:
#            self._default_get_neighs()
        return neighs

    def _get_neighs_slice(self, k_is=[0]):
        """Getting neighs from slice.

        Parameters
        ----------
        ks: slice (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.

        """
        neighs = [np.array([range(self.idxs.start, self.idxs.stop,
                                  self.idxs.step)
                            for j in range(len(self.iss))])
                  for i in range(len(k_is))]
        neighs = np.array(neighs)
        return neighs

    def _get_neighs_array_dynamic(self, k_is=[0]):
        """Getting neighs from array.

        Parameters
        ----------
        ks: np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.

        """
        neighs = self.idxs[k_is, :, :]
        return neighs

    def _get_neighs_array_static(self, k_is=[0]):
        """Getting neighs from array.

        Parameters
        ----------
        ks: np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.

        """
        neighs = [self.idxs for i in range(len(k_is))]
        neighs = np.array(neighs)
        return neighs

    def _get_neighs_list_dynamic(self, k_is=[0]):
        """Getting neighs from list.

        Parameters
        ----------
        ks: list (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.

        """
        neighs = [self.idxs[k_i] for k_i in k_is]
        return neighs

    def _get_neighs_list_static(self, k_is=[0]):
        """Getting neighs from list.

        Parameters
        ----------
        ks: list or np.ndarray (default=[0])
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.

        """
        neighs = [self.idxs for k_i in k_is]
        return neighs

    def _default_get_neighs(self, k_i=0):
        """Default get neighs (when it is not set)

        Parameters
        ----------
        ks: int, list or np.ndarray (default=0)
            the perturbations indices associated with the returned information.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k` required in the input.

        """
        raise Exception("Information not set in pst.Neighs_Info.")

    ########################## Get by coreinfo by iss #########################
    ## Get the neighs_info copy object with same information but iss reduced.
    ## Format into get_copy_iss and get_copy_iss_by_ind
    def _staticneighs_get_copy_iss(self, iss):
        """Get the neighs_info copy object with same information but iss
        reduced.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        Returns
        -------
        neighs_info: pst.Neighs_Info
            the neighbourhood information of the elements `i` for the
            perturbations `k`.

        """
        inds = self._get_indices_from_iss(iss)
        return self._staticneighs_get_copy_iss_by_ind(inds)

    def _notstaticneighs_get_copy_iss(self, iss):
        """Get the neighs_info copy object with same information but iss
        reduced.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        Returns
        -------
        neighs_info: pst.Neighs_Info
            the neighbourhood information of the elements `i` for the
            perturbations `k`.

        """
        inds = self._get_indices_from_iss(iss)
        return self._notstaticneighs_get_copy_iss_by_ind(inds)

    def _staticneighs_get_copy_iss_by_ind(self, indices):
        """Get the neighs_info copy object with same information but iss
        reduced.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        Returns
        -------
        neighs_info: pst.Neighs_Info
            the neighbourhood information of the elements `i` for the
            perturbations `k`.

        """
        indices = [indices] if type(indices) == int else indices
        iss = [self.iss[i] for i in indices]
        idxs, sp_relpos = self._staticneighs_get_corestored_by_inds(indices)
        ## Copy of information in new container
        neighs_info = self.copy()
        neighs_info.idxs = idxs
        neighs_info.sp_relative_pos = sp_relpos
        neighs_info.iss = iss
        return neighs_info

    def _notstaticneighs_get_copy_iss_by_ind(self, indices):
        """Get the neighs_info copy object with same information but iss
        reduced.

        Parameters
        ----------
        inds: list
            the indices of the elements codes we stored their neighbourhood.

        Returns
        -------
        neighs_info: pst.Neighs_Info
            the neighbourhood information of the elements `i` for the
            perturbations `k`.

        """
        indices = [indices] if type(indices) == int else indices
        iss = [self.iss[i] for i in indices]
        idxs, sp_relpos = self._notstaticneighs_get_corestored_by_inds(indices)
        ## Copy of information in new container
        neighs_info = self.copy()
        neighs_info.idxs = idxs
        neighs_info.sp_relative_pos = sp_relpos
        neighs_info.iss = iss
        return neighs_info

    ## Auxiliar functions
    def _staticneighs_get_corestored_by_inds_notslice(self, inds):
        """Get the neighborhood information from the indices.

        Parameters
        ----------
        inds: list
            the indices of the elements codes we stored their neighbourhood.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` and for each
            perturbation `k`.
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        inds = [inds] if type(inds) == int else inds
        idxs = [self.idxs[i] for i in inds]
        idxs = np.array(idxs) if type(self.idxs) == np.ndarray else idxs
        if self.sp_relative_pos is not None:
            sp_relative_pos = [self.sp_relative_pos[i] for i in inds]
        else:
            sp_relative_pos = None
        return idxs, sp_relative_pos

    def _notstaticneighs_get_corestored_by_inds_notslice(self, inds):
        """Get the neighborhood information from the indices.

        Parameters
        ----------
        inds: list
            the indices of the elements codes we stored their neighbourhood.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` and for each
            perturbation `k`.
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        inds = [inds] if type(inds) == int else inds
        idxs = []
        for k in range(len(self.idxs)):
            idxs.append([self.idxs[k][i] for i in inds])
        idxs = np.array(idxs) if type(self.idxs) == np.ndarray else idxs

        if self.sp_relative_pos is not None:
            sp_relative_pos = []
            for k in range(len(self.sp_relative_pos)):
                sp_relative_pos += [[self.sp_relative_pos[k][i] for i in inds]]
        else:
            sp_relative_pos = None
        return idxs, sp_relative_pos

    def _staticneighs_get_corestored_by_inds_slice(self, inds):
        """Get the neighborhood information from the indices.

        Parameters
        ----------
        inds: list
            the indices of the elements codes we stored their neighbourhood.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` and for each
            perturbation `k`.
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        inds = [inds] if type(inds) == int else inds
        idxs = self.idxs
        if self.sp_relative_pos is not None:
            sp_relative_pos = [self.sp_relative_pos[i] for i in inds]
        else:
            sp_relative_pos = None
        return idxs, sp_relative_pos

    def _notstaticneighs_get_corestored_by_inds_slice(self, inds):
        """Get the neighborhood information from the indices.

        Parameters
        ----------
        inds: list
            the indices of the elements codes we stored their neighbourhood.

        Returns
        -------
        neighs: list or np.ndarray
            the neighs information for each element `i` and for each
            perturbation `k`.
        sp_relpos: list or np.ndarray (default=None)
            the relative position information for each element `i` and for each
            perturbation `k`.

        """
        inds = [inds] if type(inds) == int else inds
        idxs = self.idxs
        if self.sp_relative_pos is not None:
            sp_relative_pos = []
            for k in range(len(self.sp_relative_pos)):
                sp_relative_pos += [[self.sp_relative_pos[k][i] for i in inds]]
        else:
            sp_relative_pos = None
        return idxs, sp_relative_pos

    def _get_indices_from_iss(self, iss):
        """Indices of iss from self.iss.

        Parameters
        ----------
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        Returns
        -------
        inds: list
            the indices of the elements codes we stored their neighbourhood.

        """
        iss = [iss] if type(iss) not in [np.ndarray, list] else iss
        if self.iss is not None:
            inds = []
            for i in iss:
                inds.append(list(self.iss).index(i))
#        else:
#            inds = iss
        return inds

    ###########################################################################
    ################################ CHECKERS #################################
    ###########################################################################
    ### Only activate that in a testing process
    def assert_goodness(self):
        """Assert standarts of storing."""
        if self._setted:
            self.assert_stored_iss()
            self.assert_stored_ks()
        ## Check idxs
        self.assert_stored_idxs()
        ## Check sp_relative_pos
        self.assert_stored_sp_rel_pos()

    def assert_stored_sp_rel_pos(self):
        """Definition of the standart store for sp_relative_pos."""
#        ## Temporal
#        if self.sp_relative_pos is not None:
#            if self._constant_neighs:
#                if self.staticneighs:
#                    assert(len(np.array(self.sp_relative_pos).shape) == 3)
#                else:
#                    assert(len(np.array(self.sp_relative_pos).shape) == 4)
#        #################
        array_types = [list, np.ndarray]
        if self.sp_relative_pos is not None:
            assert(type(self.sp_relative_pos) in [list, np.ndarray])
#            if type(self.sp_relative_pos) in [float, int, np.int32, np.int64]:
#                ### Probably redundant
#                # it is needed or possible this situation?
#                pass
            assert(type(self.sp_relative_pos) in [list, np.ndarray])
#            if self.ks is None:
#                assert(self.staticneighs)
#                assert(len(self.sp_relative_pos) == len(self.iss))
            if self.staticneighs:
                assert(len(self.sp_relative_pos) == len(self.iss))
                ## Assert deep 3
                if len(self.iss):
                    assert(type(self.sp_relative_pos[0]) in array_types)
            else:
                assert(self.ks is not None)
                assert(len(self.sp_relative_pos) == len(self.ks))
            if type(self.sp_relative_pos[0]) in array_types:
                if not self.staticneighs:
                    assert(len(self.sp_relative_pos[0]) == len(self.iss))
                if len(self.sp_relative_pos[0]) > 0:
                    assert(type(self.sp_relative_pos[0][0]) in array_types)

    def assert_stored_iss(self):
        """Definition of the standart store for iss."""
        assert(type(self.iss) == list)
        assert(len(self.iss) > 0)

    def assert_stored_ks(self):
        """Definition of the standart store for ks."""
        assert(self.ks is None or type(self.ks) in [list, np.ndarray])
        if self.ks is not None:
            assert(type(self.ks[0]) in inttypes)

    def assert_stored_idxs(self):
        """Definition of the standart store for sp_relative_pos."""
        if type(self.idxs) == list:
            assert(type(self.idxs[0]) in [list, np.ndarray])
            if not self.staticneighs:
                assert(type(self.idxs[0][0]) in [list, np.ndarray])
            else:
                if '__len__' in dir(self.idxs[0]):
                    if len(self.idxs[0]):
                        assert(type(self.idxs[0][0]) in inttypes)
                    else:
                        assert(not any(self.idxs[0]))
        elif type(self.idxs) == np.ndarray:
            if self.staticneighs:
                assert(len(self.idxs.shape) == 2)
            else:
                assert(len(self.idxs.shape) == 3)
#            if self.ks is not None and not self.staticneighs:
#                assert(len(self.idxs) == len(self.ks))
#            else:
#                assert(len(self.idxs.shape) == 2)
            if self.staticneighs:
                assert(len(self.idxs) == len(self.iss))
            else:
                assert(len(self.idxs[0]) == len(self.iss))
        elif type(self.idxs) == slice:
            pass
        else:
            ### Probably redundant (Only testing purposes)
#            print type(self.idxs), self.idxs
            types = str(type(self.idxs))
            raise Exception("Not proper type in self.idxs. Type: %s." % types)

    def check_output_standards(self, neighs, sp_relative_pos, ks, iss):
        """Check output standarts.

        Parameters
        ----------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k`.
        sp_relpos: list or np.ndarray
            the relative position information for each element `i` for each
            perturbation `k`.
        ks: list or np.ndarray
            the perturbations indices associated with the returned information.
        iss: list or np.ndarray
            the indices of the elements we stored their neighbourhood.

        """
        self.check_output_neighs(neighs, ks)
        self.check_output_rel_pos(sp_relative_pos, ks)
        assert(len(iss) == len(self.iss))

    def check_output_neighs(self, neighs, ks):
        """Check standart outputs of neighs.

        Parameters
        ----------
        neighs: list or np.ndarray
            the neighs information for each element `i` for each possible
            perturbation `k`.
        ks: list or np.ndarray
            the perturbations indices associated with the returned information.

        """
        if type(neighs) == list:
            assert(len(neighs) == len(ks))
            #assert(type(neighs[0]) == list)
            assert(len(neighs[0]) == len(self.iss))
        elif type(neighs) == np.ndarray:
            assert(len(neighs.shape) == 3)
            assert(len(neighs) == len(ks))
            assert(neighs.shape[1] == len(self.iss))
        else:
            ### Probably redundant (Only testing purposes)
#            print neighs
            types = str(type(neighs))
            raise Exception("Not correct neighs output.Type: %s." % types)

    def check_output_rel_pos(self, sp_relative_pos, ks):
        """Check standart outputs of rel_pos.

        Parameters
        ----------
        sp_relpos: list or np.ndarray
            the relative position information for each element `i` for each
            perturbation `k`.
        ks: list or np.ndarray
            the perturbations indices associated with the returned information.

        """
        assert(type(sp_relative_pos) in [np.ndarray, list])
        assert(len(sp_relative_pos) == len(ks))
        assert(len(sp_relative_pos[0]) == len(self.iss))

    ########################### Joinning functions ############################
    def _format_joining_functions(self):
        """Format the joining functions to use."""
        ## TODO: Extend to n possible neighs_info elements
        if self.staticneighs:
            if self.ifdistance:
                self.join_neighs_and = join_neighsinfo_AND_static_dist
                self.join_neighs_or = join_neighsinfo_OR_static_dist
                self.join_neighs_xor = join_neighsinfo_XOR_static_dist
            else:
                self.join_neighs_and = join_neighsinfo_AND_static_notdist
                self.join_neighs_or = join_neighsinfo_OR_static_notdist
                self.join_neighs_xor = join_neighsinfo_XOR_static_notdist
        else:
            if self.ifdistance:
                self.join_neighs_and = join_neighsinfo_AND_notstatic_dist
                self.join_neighs_or = join_neighsinfo_OR_notstatic_dist
                self.join_neighs_xor = join_neighsinfo_XOR_notstatic_dist
            else:
                self.join_neighs_and = join_neighsinfo_AND_notstatic_notdist
                self.join_neighs_or = join_neighsinfo_OR_notstatic_notdist
                self.join_neighs_xor = join_neighsinfo_XOR_notstatic_notdist

    def join_neighs(self, neighs_info, mode='and', joiner_pos=None):
        """General joining function.

        Parameters
        ----------
        neighs_info: pst.Neighs_Info
            the neighbourhood information of the other neighs we want to join.
        mode: str optional ['and', 'or', 'xor']
            the type of joining process we want to do.
        joiner_pos: function (default=None)
            the function to join the relative positions of the different
            neighbourhood.

        Returns
        -------
        new_neighs_info: pst.Neighs_Info
            the neighbourhood information of joined neighbourhood.

        """
        assert(mode in ['and', 'or', 'xor'])
        if mode == 'and':
            if self.ifdistance:
                new_neighs_info = self.join_neighs_and(self, neighs_info,
                                                       joiner_pos)
            else:
                new_neighs_info = self.join_neighs_and(self, neighs_info)
        elif mode == 'or':
            if self.ifdistance:
                new_neighs_info = self.join_neighs_or(self, neighs_info,
                                                      joiner_pos)
            else:
                new_neighs_info = self.join_neighs_or(self, neighs_info)
        elif mode == 'xor':
            if self.ifdistance:
                new_neighs_info = self.join_neighs_xor(self, neighs_info,
                                                       joiner_pos)
            else:
                new_neighs_info = self.join_neighs_xor(self, neighs_info)
        return new_neighs_info


###############################################################################
######################### Auxiliar inspect functions ##########################
###############################################################################
def ensuring_neighs_info(neighs_info, k):
    """Ensuring that the neighs_info is in Neighs_Info object container.

    Parameters
    ----------
    neighs_info: pst.Neighs_Info or tuple
        the neighbourhood information.
    k: list
        the list of perturbation indices.

    Returns
    -------
    neighs_info: pst.Neighs_Info
        the properly formatted neighbourhood information.

    """
    if not type(neighs_info).__name__ == 'instance':
        parameters = inspect_raw_neighs(neighs_info, k=k)
        parameters['format_structure'] = 'tuple_k'
        neighs_info_object = Neighs_Info(**parameters)
        neighs_info_object.set((neighs_info, k))
        neighs_info = neighs_info_object
    return neighs_info


def inspect_raw_neighs(neighs_info, k=0):
    """Useful class to inspect a raw structure neighs, in order to set
    some parts of the class in order to a proper settting adaptation.

    Parameters
    ----------
    neighs_info: pst.Neighs_Info or tuple
        the neighbourhood information.
    k: int or list (default=0)
        the list of perturbation indices.

    Returns
    -------
    parameters: dict
        the parameters to reinstantiate the neighbourhood information
        properly.

    """
    deep = find_deep(neighs_info)
    k = [k] if type(k) == int else k
    parameters = {'format_structure': 'raw'}
    parameters['format_level'] = deep
    if deep == 3:
        assert(np.max(k) <= len(neighs_info))
        parameters['kret'] = len(neighs_info)
        parameters['staticneighs'] = False
    else:
        parameters['staticneighs'] = True
        parameters['kret'] = np.max(k)
    return parameters


def find_deep(neighs_info):
    """Find deep from a raw structure.

    Parameters
    ----------
    neighs_info: tuple
        the neighbourhood information.

    Returns
    -------
    deep: int
        the level in which the information is provided.

    """
    if '__len__' not in dir(neighs_info):
        deep = 0
    else:
        if len(neighs_info) == 0:
            deep = 1
        elif '__len__' not in dir(neighs_info[0]):
            deep = 1
        else:
            logi = [len(neighs_info[i]) == 0 for i in range(len(neighs_info))]
            if all(logi):
                deep = 2
            elif '__len__' not in dir(neighs_info[0][0]):
                deep = 2
            else:
                deep = 3
    return deep


def neighsinfo_features_preformatting_tuple(key, k_perturb):
    """Preformatting tuple.

    Parameters
    ----------
    neighs_info: tuple
        the neighborhood information. Assumed that tuple input:
            * idxs, ks
    k_perturb: int
        the number of perturbations.

    Returns
    -------
    neighs: list or np.ndarray
        the neighs information for each element `i` for each possible
        perturbation `k`.
    ks: list or np.ndarray
        the perturbations indices associated with the returned information.
    sp_relpos: list or np.ndarray
        the relative position information for each element `i` for each
        perturbation `k`.

    """
    deep = find_deep(key[0])
    if deep == 1:
        ks = [key[1]] if type(key[1]) == int else key[1]
        i, k, d = neighsinfo_features_preformatting_list(key[0], ks)
    else:
        neighs_info = Neighs_Info()
        neighs_info.set_information(k_perturb)
        neighs_info.set(key)
        # Get information
        i, d, k, _ = neighs_info.get_information()
    return i, k, d


def neighsinfo_features_preformatting_list(key, k_perturb):
    """Preformatting list.

    Parameters
    ----------
    neighs_info: list
        the neighborhood information. Assumed that tuple input:
            * idxs, ks
    k_perturb: int
        the number of perturbations.

    Returns
    -------
    neighs: list or np.ndarray
        the neighs information for each element `i` for each possible
        perturbation `k`.
    ks: list or np.ndarray
        the perturbations indices associated with the returned information.
    sp_relpos: list or np.ndarray
        the relative position information for each element `i` for each
        perturbation `k`.

    """
    kn = range(k_perturb+1) if type(k_perturb) == int else k_perturb
    key = [[idx] for idx in key]
    i, k, d = np.array([key]*len(kn)), kn, [[None]*len(key)]*len(kn)
    return i, k, d


###############################################################################
####################### Complementary Joinning function #######################
###############################################################################
def join_by_iss(list_neighs_info):
    """Joinning by iss.

    Parameters
    ----------
    list_neighs_info: list of pst.Neighs_Info
        the list of different neighbourhood information, with overlapping
        set of iss.

    Returns
    -------
    neighs_info: tuple
        the joined neighbourhood information.

    """
    ## Computation
    if len(list_neighs_info) == 1:
        return list_neighs_info[0]
    static = list_neighs_info[0].staticneighs
    ifdistance = list_neighs_info[0].sp_relative_pos is not None
    assert([nei.sp_relative_pos == ifdistance for nei in list_neighs_info])
    assert([nei.staticneighs == static for nei in list_neighs_info])
    ks = list_neighs_info[0].ks
#    print ks
#    print [nei.ks for nei in list_neighs_info]
    assert(all([len(nei.ks) == len(ks) for nei in list_neighs_info]))
    assert(all([nei.ks == ks for nei in list_neighs_info]))
    if static:
        sp_relative_pos = None if not ifdistance else []
        iss, idxs = [], []
        for nei in list_neighs_info:
            if type(nei.idxs) != slice:
                idxs += list(nei.idxs)
            else:
                idxs.append(nei.idxs)
            iss += nei.iss
            if ifdistance:
                sp_relative_pos += list(nei.sp_relative_pos)
    else:
        sp_relative_pos = None if not ifdistance else []
        iss = list(np.hstack([nei.iss for nei in list_neighs_info]))
        idxs = []
        for k in range(len(ks)):
            idxs_k = []
            sp_relative_pos_k = None if not ifdistance else []
            for nei in list_neighs_info:
                idxs_k += list(nei.idxs[k])
                if ifdistance:
                    sp_relative_pos_k += list(nei.sp_relative_pos[k])
            idxs.append(idxs_k)
            if ifdistance:
                sp_relative_pos.append(sp_relative_pos_k)
    constant = list_neighs_info[0]._constant_neighs
    assert([nei._constant_neighs == constant for nei in list_neighs_info])
    if constant:
        idxs = np.array(idxs)

    ## Formatting
    level = 2 if static else 3
    _, type_neighs, type_sp_rel_pos, _ = list_neighs_info[0].format_set_info
    format_get_info, format_get_k_info = list_neighs_info[0].format_get_info
    type_neighs = 'array' if constant else 'list'

    nei = Neighs_Info(constant_neighs=constant, format_structure='tuple_only',
                      format_get_info=None, format_get_k_info=None,
                      format_set_iss='list', staticneighs=static,
                      ifdistance=ifdistance, type_neighs=type_neighs,
                      format_level=level)
    neighs_nfo = (idxs, sp_relative_pos) if ifdistance else (idxs,)
    nei.set(neighs_nfo, iss)
    nei.set_ks(ks)
    return nei
