
"""
neighs information
------------------
Auxiliar class in order to manage the information of the neighbourhood
returned by the retrievers.
Due to the difficulty of the structure it is convenient to put altogether
in a single class and manage better in a centralized way all the different
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

"""

import numpy as np
import warnings
warnings.filterwarnings("always")


class Neighs_Info:
    """Class to store, move and manage the neighbourhood information retrieved.
    """
    type_ = "pySpatialTools.Neighs_Info"

    def __init__(self, constant_neighs=False, kret=1, format_structure=None,
                 n=0, format_get_info=None, format_get_k_info=None,
                 format_set_iss=None, staticneighs=None, ifdistance=None,
                 type_neighs=None, type_sp_rel_pos=None, format_level=None):
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

    def __iter__(self):
        """Get information sequentially."""
        for i in range(len(self.ks)):
            yield self.get_neighs([i]), self.get_sp_rel_pos([i]),\
                [self.ks[i]], self.iss

    def empty(self):
        return not self.any()

    def any(self):
        boolean = True
        if type(self.idxs) == np.ndarray:
            boolean = all(self.idxs.shape)
        elif type(self.idxs) == list:
            sh = np.array(self.idxs).shape
            if len(sh) >= 2:
                boolean = np.all(sh)
        return boolean

    def reset(self):
        self._set_init()

    @property
    def shape(self):
        """Return the number of indices, neighbours and ks considered. For
        irregular cases the neighbours number is set as None."""
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
        """Set specific global information."""
        self._n = n
        self._kret = k_perturb

    def reset_functions(self):
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
        _, aux1, aux2, aux3 = self.format_set_info
        self.format_set_info = format_structure, aux1, aux2, aux3
        self.reset_format()

    def reset_level(self, format_level):
        self.level = format_level
        self.reset_format()

    def reset_format(self):
        ## Formatters
        self._format_setters(*self.format_set_info)
        self._format_getters(*self.format_get_info)

    def set_types(self, type_neighs=None, type_sp_rel_pos=None):
        ## 1. Set set_sp_rel_pos
        self.type_neighs, self.type_sp_rel_pos = type_neighs, type_sp_rel_pos

        if self.ifdistance is False:
            self.set_sp_rel_pos = self._null_set_rel_pos
            self.get_sp_rel_pos = self._null_get_rel_pos
        else:
            self.get_sp_rel_pos = self._general_get_rel_pos
            if self.level < 2:
                self.get_sp_rel_pos = self._static_get_rel_pos
            if type_sp_rel_pos is None:
                self.set_sp_rel_pos = self._general_set_rel_pos
            elif type_sp_rel_pos == 'general':
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
                    self.set_sp_rel_pos = self._array_only_set_rel_pos
                elif self.level == 2:
                    self.set_sp_rel_pos = self._array_array_set_rel_pos
                elif self.level == 3:
                    self.set_sp_rel_pos = self._array_array_array_set_rel_pos

        ## 2. Set set_neighs
        if type_neighs is None:
            self.set_neighs = self._general_set_neighs
        elif type_neighs == 'general':
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
        elif format_structure == 'tuple_tuple':
            self._set_info = self._set_tuple_tuple_structure
        elif format_structure == 'list_tuple_only':
            assert(self.level == 2)
            self._set_info = self._set_list_tuple_only_structure
            self.staticneighs_set = False
            if self.level != 2:
                raise Exception("Not correct inputs.")
            else:
                self.level = 3
        elif format_structure == 'list_tuple':
            assert(self.level == 2)
            self._set_info = self._set_list_tuple_structure
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
        other information and functions."""
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

        ## 5. General set
        self.set = self._general_set

    def _format_set_iss(self, format_set_iss=None):
        ## Format iss
        if format_set_iss is None:
            self._set_iss = self._general_set_iss
        elif format_set_iss == 'general':
            self._set_iss = self._general_set_iss
        elif format_set_iss == 'null':
            self._set_iss = self._null_set_iss
        elif format_set_iss == 'int':
            self._set_iss = self._int_set_iss
        elif format_set_iss == 'list':
            self._set_iss = self._list_set_iss

    def _format_getters(self, format_get_info=None, format_get_k_info=None):
        """Function to program this class according to the stored idxs."""
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

    def _postformat(self):
        """Format properly."""
        self._main_postformat()
        self._iss_postformat()
        self._assert_ks_postformat()
        self._idxs_postformat()

    def _cte_postformat(self):
        """To array because of constant neighs."""
        if type(self.idxs) == list:
            self.idxs = np.array(self.idxs)
        if self.sp_relative_pos is not None:
            self.sp_relative_pos = np.array(self.sp_relative_pos)

    def _assert_iss_postformat(self):
        if type(self.idxs) in [list, np.ndarray]:
#            print self.idxs, self.iss, self.set_neighs
            if self.staticneighs:
                if len(self.idxs) != len(self.iss):
                    assert(len(self.idxs[0]) == len(self.iss))
                    self.idxs = self.idxs[0]
            else:
                assert(all([len(k) == len(self.idxs[0]) for k in self.idxs]))

    def _assert_ks_postformat(self):
        if type(self.idxs) in [list, np.ndarray]:
            if self.ks is None:
                self.ks = range(len(self.idxs))
            if self.staticneighs:
                pass
            else:
#                print self.ks, self.idxs, self.set_neighs
                assert(len(self.ks) == len(self.idxs))
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

#    def _array_ele_postformat(self, ele):
#        return np.array(ele)
#
#    def _null_ele_postformat(self, ele):
#        return ele

    def _null_postformat(self):
        """Not change anything."""
        pass

    def _idxs_postformat_array(self):
        """"""
        self.idxs = np.array(self.idxs)

    def _idxs_postformat_null(self):
        """"""
        pass

    ###########################################################################
    ################################## SETS ###################################
    ###########################################################################

    ########################### Setters candidates ############################
    ###########################################################################
    def _general_set(self, neighs_info, iss=None):
        """General set."""
#        print 'set neighs_info:'+'.'*20
#        print neighs_info
#        print self.staticneighs
        self._preset(neighs_info, iss)
        ## Post-set functions
#        print self.staticneighs, type(self.idxs), self.idxs
        if type(self.idxs) == np.ndarray:
            pass
#            print '='*10, self.idxs.shape
        if type(self.idxs) == slice:
            self.get_neighs = self._get_neighs_slice
#            self.ks = list(range(self._kret+1))
        elif type(self.idxs) == np.ndarray:
            if len(self.idxs.shape) == 3:
                self.ks = list(range(len(self.idxs)))
            else:
                self.staticneighs_set = True
            if self.staticneighs:
                self.get_neighs = self._get_neighs_array_static
            else:
                self.get_neighs = self._get_neighs_array_dynamic
        elif type(self.idxs) == list:
            if self.staticneighs:
                self.get_neighs = self._get_neighs_list_static
            else:
                self.get_neighs = self._get_neighs_list_dynamic
#            self.ks = list(range(self._kret+1))
        self.assert_goodness()

    def _preset(self, neighs_info, iss=None):
        """Set the class."""
#        print '=p'*20, self.staticneighs, neighs_info
        self._reset_stored()
        self._set_iss(iss)
#        print self.staticneighs, self.set_neighs, self._set_info
        self._set_info(neighs_info)
#        print self.staticneighs, self.set_neighs
        self._postformat()
#        print self.staticneighs, self.set_neighs

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
        neighs_info:
            * (i, k)
            * (neighs, k)
            * (neighs_info, k)
                where neighs_info is a tuple which could contain (neighs,
                dists) or (neighs,)
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

    ############################## Set Structure ##############################
    ###########################################################################
    def _set_raw_structure(self, key):
        """Raw structure.
        * neighs{any form}
        """
        self.set_neighs(key)
        self.ifdistance = False

    def _set_structure_tuple(self, key):
        """Tuple general.
        """
        if len(key) == 2:
            msg = "Ambiguous input in `set` function of pst.Neighs_Info."
            warnings.warn(msg, SyntaxWarning)
            if type(key[0]) == tuple:
                self._set_structure_tuple(key[0])
                self.ks = list(np.array([key[1]]).ravel())
            else:
                aux_bool = type(key[0]) in [np.ndarray, list]
                if type(key[0]) == type(key[1]) and aux_bool:
                    if len(key[0]) == len(key[1]):
                        self._set_tuple_only_structure(key)
                    else:
                        self.set_neighs(key[0])
                        self.ks = list(np.array(key[1]))
                else:
                    self._set_tuple_only_structure(key)
        else:
            self.set_neighs(key[0])

    def _set_tuple_structure(self, key):
        """Tuple structure.
        * (neighs_info{any form}, ks)
        """
        if len(key) == 2:
            self.ks = list(np.array(key[1]))
        self.set_neighs(key[0])

    def _set_tuple_only_structure(self, key):
        """Tuple only structure.
        * (neighs{any form}, sp_relative_pos{any form})
        """
        self.set_neighs(key[0])
        if len(key) == 2:
            self.set_sp_rel_pos(key[1])
        elif len(key) > 2:
            raise TypeError("Not correct input.")

    def _set_tuple_tuple_structure(self, key):
        """Tuple tuple structure.
        * ((neighs{any form}, sp_relative_pos{any form}), ks)
        """
        if len(key) == 2:
            ks = [key[1]] if type(key[1]) == int else key[1]
            self.ks = list(np.array([ks]).ravel())
        self._set_tuple_only_structure(key[0])

    def _set_tuple_list_tuple_only(self, key):
        """
        * (neighs_info{list of typle only}, ks)
        """
        self.ks = list(np.array(key[1]))
        self._set_list_tuple_only_structure(key[0])

    def _set_structure_list(self, key):
        """General list structure.
        * [neighs_info{tuple form}]
        """
        if type(key[0]) == tuple:
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
            self.set.neighs(np.array(key))
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
        """List tuple only structure.
        * [(neighs{any form}, sp_relative_pos{any form})]
        """
        ## Change to list and whatever it was
        self.set_neighs([e[0] for e in key])
        self.set_sp_rel_pos([e[1] for e in key])

    def _set_list_tuple_structure(self, key):
        ks = [key[1]] if type(key[1]) == int else key[1]
        print ks, key[0], type(key[1]), key
        assert(len(key[0]) == len(ks))
        self._set_list_tuple_only_structure(key[0])
        self.ks = ks

    ############################### Set Neighs ################################
    ###########################################################################
    def _general_set_neighs(self, key):
        """General setting of only neighs.
        * neighs {number form}
        * neighs {list form}
        * neighs {array form}
        """
        if type(key) == list:
            self._set_neighs_general_list(key)
        elif type(key) == np.ndarray:
            self._set_neighs_general_array(key)
        elif type(key) in [int, np.int32, np.int64]:
            self._set_neighs_number(key)
        else:
            print key
            raise TypeError("Incorrect neighs input in pst.Neighs_Info")

    def _set_neighs_number(self, key):
        """Only one neighbor expressed in a number way.
        * indice{int form}
        """
        if self.staticneighs:
            self.idxs = np.array([[key]]*len(self.iss))
        else:
            len_ks = 1 if self.ks is None else len(self.ks)
            self.idxs = np.array([[[key]]*len(self.iss)]*len_ks)
        self._constant_neighs = True
        self._setted = True

    def _set_neighs_slice(self, key):
        """
        * indices{slice form}
        """
        self._constant_neighs = True
        if key is None:
            self.idxs = slice(0, self._n, 1)
        elif isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = self._n if key.stop is None else key.stop
            stop = self._n if key.stop > 10*16 else key.stop
            step = 1 if key.step is None else key.step
            self.idxs = slice(start, stop, step)
        self._setted = True

    def _set_neighs_array_lvl1(self, key):
        """
        * indices{np.ndarray form} shape: (neighs)
        """
        #sh = key.shape
        ## If only array of neighs
        if self.staticneighs:
            self.idxs = np.array([key for i in range(len(self.iss))])
        else:
            len_ks = len(self.ks) if self.ks is not None else 1
            self.idxs = np.array([[key for i in range(len(self.iss))]
                                  for i in range(len_ks)])
        self._setted = True

    def _set_neighs_array_lvl2(self, key):
        """
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
        """
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
        """
        * indices{np.ndarray form} shape: (neighs)
        * indices{np.ndarray form} shape: (iss, neighs)
        * indices{np.ndarray form} shape: (ks, iss, neighs)
        """
        inttypes = [int, np.int32, np.int64]
        key = np.array([key]) if type(key) in inttypes else key
        sh = key.shape
        ## If only array of neighs
        if len(sh) == 0:
            self._setted = False
            if self.staticneighs:
                self.idxs = np.array([[]])
            else:
                self.idxs = np.array([[[]]])
        elif len(sh) == 1:
            self._set_neighs_array_lvl1(key)
        ## If only iss and neighs
        elif len(sh) == 2:
            self._set_neighs_array_lvl2(key)
        elif len(sh) == 3:
            self._set_neighs_array_lvl3(key)

    def _set_neighs_general_list(self, key):
        """
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
        """
        * indices {list of list form [neighs]} [neighs]
        """
        self._set_neighs_array_lvl1(np.array(key))

    def _set_neighs_list_list(self, key):
        """
        * [neighs_info{array-like form}, ...] [iss][neighs]
        """
        if self._constant_neighs:
            key = np.array(key)
        if self.staticneighs:
            self.idxs = key
            self.ks = range(1) if self.ks is None else self.ks
        else:
            len_ks = 1 if self.ks is None else len(self.ks)
            self.idxs = [key for k in range(len_ks)]
            self.ks = range(1) if self.ks is None else self.ks
            if type(key) == np.ndarray:
                self.idxs = np.array(self.idxs)
        if len(self.iss) != len(key):
            self.iss = range(len(key))
#        if len(self.idxs[0]) > 0:
#            self.iss = list(range(len(self.idxs)))
        self._setted = True

    def _set_neighs_list_list_list(self, key):
        """
        * [neighs_info{array-like form}, ...] [ks][iss][neighs]
        """
        self.ks = list(range(len(key)))
        if self._constant_neighs:
            self.idxs = np.array(key)
        else:
            self.idxs = key
        if len(self.idxs[0]):
            self.iss = list(range(len(self.idxs[0])))
        if self.staticneighs:
            self.idxs = self.idxs[0]
        self._setted = True

    ########################### Set Sp_relative_pos ###########################
    ###########################################################################
    def _general_set_rel_pos(self, rel_pos):
        """
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
            print rel_pos
            msg = "Incorrect relative position input in pst.Neighs_Info"
            raise TypeError(msg)

    def _set_rel_pos_general_list(self, rel_pos):
        """
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
                self._list_only_set_rel_pos(rel_pos)
            elif type(rel_pos[0]) not in [list, np.ndarray]:
                self._list_only_set_rel_pos(rel_pos)
            else:
                if len(rel_pos[0]) == 0:
                    self._list_list_only_set_rel_pos(rel_pos)
                elif type(rel_pos[0][0]) in [list, np.ndarray]:
                    self._list_list_only_set_rel_pos(rel_pos)
                else:
                    self._list_list_set_rel_pos(rel_pos)

    def _null_set_rel_pos(self, rel_pos):
        """Not consider the input."""
        self.get_sp_rel_pos = self._null_get_rel_pos

    def _set_rel_pos_number(self, rel_pos):
        """Number set pos."""
        self.sp_relative_pos = self._set_rel_pos_dim([rel_pos])

    def _set_rel_pos_dim(self, rel_pos):
        """Set rel pos.
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
        """Array only. [nei][dim] or [nei]"""
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
        """Array or arrays. [iss][nei][dim] or [nei]."""
#        self.staticneighs = True
        if self.staticneighs:
            self.sp_relative_pos = np.array(rel_pos)
        else:
            len_ks = 1 if self.ks is None else len(self.ks)
            self.sp_relative_pos = np.array([rel_pos for k in range(len_ks)])

    def _array_array_array_set_rel_pos(self, rel_pos):
        """Array or arrays. [ks][iss][nei][dim] or [ks][nei]."""
        if self.staticneighs:
            self.sp_relative_pos = rel_pos[0]
        else:
            self.sp_relative_pos = rel_pos

    def _list_only_set_rel_pos(self, rel_pos):
        """List only relative pos. Every iss and ks has the same neighs with
        the same relative information. [nei][dim] or [nei]
        """
        self._array_only_set_rel_pos(rel_pos)

    def _list_list_only_set_rel_pos(self, rel_pos):
        """List list only relative pos. Every ks has the same neighs with the
        same relative information. [iss][nei][dim] or [iss][nei]
        """
        if self.staticneighs is not True:
            assert(self.ks is not None)
            n_ks = len(self.ks)
            self.sp_relative_pos = [rel_pos]*n_ks
        else:
            self.sp_relative_pos = rel_pos

    def _list_list_set_rel_pos(self, rel_pos):
        if self.staticneighs:
            self.sp_relative_pos = rel_pos[0]
        else:
            self.sp_relative_pos = rel_pos

    ############################### Setter iss ################################
    ###########################################################################
    def _general_set_iss(self, iss):
        """General set iss input."""
        if type(iss) == int:
            self._int_set_iss(iss)
        elif type(iss) in [list, np.ndarray]:
            self._list_set_iss(iss)
        else:
            if type(self.idxs) in [list, np.ndarray]:
                if self.staticneighs:
                    self.iss = range(len(self.idxs))
                else:
                    if len(self.idxs[0]):
                        self.iss = range(len(self.idxs[0]))

    def _int_set_iss(self, iss):
        """Input iss always integer."""
        self.iss = [iss]

    def _list_set_iss(self, iss):
        """Input iss always array-like."""
        self.iss = list(iss)

    def _null_set_iss(self, iss):
        """Not consider the input."""
        pass

    ###########################################################################
    ################################## GETS ###################################
    ###########################################################################

    ############################# Getter rel_pos ##############################
    ###########################################################################
    def _general_get_rel_pos(self, k_is=[0]):
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
        return np.array([self.sp_relative_pos for k in k_is])

#    def _static_rel_pos_list(self, k_is=[0]):
#        return self.sp_relative_pos*len(k_is)
#
#    def _static_rel_pos_array(self, k_is=[0]):
#        return np.array([self.sp_relative_pos for i in range(len(k_is))])

    def _dynamic_rel_pos_list(self, k_is=[0]):
#        [[e[k_i] for e in self.sp_relative_pos] for k_i in k_is]
        return [self.sp_relative_pos[i] for i in k_is]

    def _dynamic_rel_pos_array(self, k_is=[0]):
#        [[e[k_i] for e in self.sp_relative_pos] for k_i in k_is]
        return np.array([self.sp_relative_pos[i] for i in k_is])

    ################################ Getters k ################################
    ###########################################################################
    def _general_get_k(self, k=None):
        """General get k."""
        ## Format k
        if k is None:
            ks = self._default_get_k()
        elif type(k) in [np.ndarray, list]:
            ks = self._list_get_k(k)
        elif type(k) in [int, np.int32, np.int64]:
            ks = self._integer_get_k(k)
        return ks

    def _default_get_k(self, k=None):
        """Default get ks."""
        return [0]

    def _integer_get_k(self, k):
        """Integer get k."""
        if type(k) == list:
            return k
        if k >= 0 or k <= self._kret:
            ks = [k]
        else:
            raise TypeError("k index out of bounds.")
        return ks

    def _list_get_k(self, k):
        """List get k."""
        ks = [self._integer_get_k(k_i)[0] for k_i in k]
        return ks

    def _get_k_indices(self, ks):
        """List of indices of ks."""
        if self.ks is None:
            idx_ks = range(len(ks))
        else:
            if self.staticneighs:
                idx_ks = ks
            else:
                idx_ks = [self.ks.index(e) for e in ks]
        return idx_ks

    ############################ Getters information ##########################
    ###########################################################################
    def _general_get_information(self, k=None):
        """Get information stored in this class."""
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
        """For the unset instances."""
        raise Exception("Information not set in pst.Neighs_Info.")

    def _get_neighs_general(self, k_is=[0]):
        """General getting neighs."""
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
        """Getting neighs from slice."""
        neighs = [np.array([range(self.idxs.start, self.idxs.stop,
                                  self.idxs.step)
                            for j in range(len(self.iss))])
                  for i in range(len(k_is))]
        neighs = np.array(neighs)
        return neighs

    def _get_neighs_array_dynamic(self, k_is=[0]):
        """Getting neighs from array."""
        neighs = self.idxs[k_is, :, :]
        return neighs

    def _get_neighs_array_static(self, k_is=[0]):
        """Getting neighs from array."""
        neighs = [self.idxs for i in range(len(k_is))]
        neighs = np.array(neighs)
        return neighs

    def _get_neighs_list_dynamic(self, k_is=[0]):
        """Getting neighs from list."""
        neighs = [self.idxs[k_i] for k_i in k_is]
        return neighs

    def _get_neighs_list_static(self, k_is=[0]):
        """Getting neighs from list."""
        neighs = [self.idxs for k_i in k_is]
        return neighs

    def _default_get_neighs(self, k_i=0):
        """Default get neighs (when it is not set)"""
        raise Exception("Information not set in pst.Neighs_Info.")

    ###########################################################################
    ################################ CHECKERS #################################
    ###########################################################################
    ### Only activate that in a testing process
    def assert_goodness(self):
        """Assert standarts of storing."""
        #print 'assert_stored', self.idxs, self.sp_relative_pos, self.iss
        #print self.ks, self.staticneighs, self._constant_neighs
        if self._setted:
            self.assert_stored_iss()
            self.assert_stored_ks()
        ## Check idxs
        self.assert_stored_idxs()
        ## Check sp_relative_pos
        self.assert_stored_sp_rel_pos()

    def assert_stored_sp_rel_pos(self):
        """Definition of the standart store for sp_relative_pos."""
        ## Temporal
        #print self.sp_relative_pos
        #print self.sp_relative_pos, np.array(self.sp_relative_pos).shape, self.set_sp_rel_pos
        if self.sp_relative_pos is not None:
            if self._constant_neighs:
                if self.staticneighs:
                    #print np.array(self.sp_relative_pos).shape
                    #print self.sp_relative_pos
                    assert(len(np.array(self.sp_relative_pos).shape) == 3)
                else:
                    assert(len(np.array(self.sp_relative_pos).shape) == 4)
        #################
        array_types = [list, np.ndarray]
        #print self.sp_relative_pos, self.sp_relative_pos is None, type(self.sp_relative_pos), self.ifdistance
        if self.sp_relative_pos is not None:
            if type(self.sp_relative_pos) in [float, int, np.int32, np.int64]:
                pass
            else:
                assert(type(self.sp_relative_pos) in [list, np.ndarray])
                if self.ks is None:
                    assert(self.staticneighs)
                    assert(len(self.sp_relative_pos) == len(self.iss))
                if self.staticneighs:
                    #print self.sp_relative_pos, self.iss
                    assert(len(self.sp_relative_pos) == len(self.iss))
                    ## Assert deep 3
                    if len(self.iss):
                        assert(type(self.sp_relative_pos[0]) in array_types)
                else:
                    assert(len(self.sp_relative_pos) == len(self.ks))
                if type(self.sp_relative_pos[0]) in array_types:
                    if not self.staticneighs:
                        assert(len(self.sp_relative_pos[0]) == len(self.iss))
                    if len(self.sp_relative_pos[0]) > 0:
                        assert(type(self.sp_relative_pos[0][0]) in array_types)
                else:
                    pass

    def assert_stored_iss(self):
        """Definition of the standart store for iss."""
        assert(type(self.iss) == list)
        #print self.iss, self.idxs, self.ks, self.sp_relative_pos
        assert(len(self.iss) > 0)

    def assert_stored_ks(self):
        """Definition of the standart store for ks."""
        assert(self.ks is None or type(self.ks) in [list, np.ndarray])
        if self.ks is not None:
            assert(type(self.ks[0]) in [int, np.int32, np.int64])

    def assert_stored_idxs(self):
        """Definition of the standart store for sp_relative_pos."""
#        print self.idxs, type(self.idxs)
        int_listtypes = [int, np.int32, np.int64]
        if type(self.idxs) == list:
#            print self.idxs, 'assertion', self.set_structure, self.set_neighs, self.format_set_info
            assert(type(self.idxs[0]) in [list, np.ndarray])
            if not self.staticneighs:
#                print self.idxs, self.ks, self.iss
                assert(type(self.idxs[0][0]) in [list, np.ndarray])
            else:
#                print self.staticneighs, self.set_neighs, self.idxs
                if '__len__' in dir(self.idxs[0]):
                    if len(self.idxs[0]):
                        assert(type(self.idxs[0][0]) in int_listtypes)
                    else:
                        assert(not any(self.idxs[0]))
        elif type(self.idxs) == np.ndarray:
#            print self.idxs.shape, self.idxs, self.set_neighs, self._idxs_postformat
#            print self._constant_neighs
            if self.staticneighs:
                assert(len(self.idxs.shape) == 2)
            else:
                assert(len(self.idxs.shape) == 3)
#            if self.ks is not None and not self.staticneighs:
##                print self.idxs.shape, len(self.ks), len(self.iss)
#                assert(len(self.idxs) == len(self.ks))
#            else:
#                print self.idxs, self.staticneighs, self.ks
#                assert(len(self.idxs.shape) == 2)
#            print self.idxs, self.iss
            if self.staticneighs:
                assert(len(self.idxs) == len(self.iss))
            else:
#                print self.iss, self.idxs, self.ks
                assert(len(self.idxs[0]) == len(self.iss))
        elif type(self.idxs) == slice:
            pass
        else:
            print type(self.idxs), self.idxs
            raise Exception("Not proper type in self.idxs.")

    def check_output_standards(self, neighs, sp_relative_pos, ks, iss):
        """Check output standarts."""
        self.check_output_neighs(neighs, ks)
        self.check_output_rel_pos(sp_relative_pos, ks)
        assert(len(iss) == len(self.iss))

    def check_output_neighs(self, neighs, ks):
        """Check standart outputs of neighs."""
#        print neighs, ks, self.idxs, self.iss
        if type(neighs) == list:
            assert(len(neighs) == len(ks))
            #assert(type(neighs[0]) == list)
            #print neighs, self.iss, self._constant_neighs, self.staticneighs
            assert(len(neighs[0]) == len(self.iss))
        elif type(neighs) == np.ndarray:
#            print neighs, self.staticneighs
            assert(len(neighs.shape) == 3)
            assert(len(neighs) == len(ks))
#            print len(self.iss)
#            print neighs.shape
            assert(neighs.shape[1] == len(self.iss))
        else:
            print neighs
            raise Exception("Not correct neighs output.")

    def check_output_rel_pos(self, sp_relative_pos, ks):
        #print sp_relative_pos, self.sp_relative_pos
        assert(type(sp_relative_pos) in [np.ndarray, list])
#        print self.sp_relative_pos, self.staticneighs, sp_relative_pos
#        print np.array(sp_relative_pos).shape
#        print np.array(sp_relative_pos).shape, ks, self.iss
#        print np.array(self.sp_relative_pos).shape, np.array(self.idxs).shape
#        print self.set_neighs, self.set_structure, self.set_sp_rel_pos
#        print self.staticneighs
        assert(len(sp_relative_pos) == len(ks))
        assert(len(sp_relative_pos[0]) == len(self.iss))
