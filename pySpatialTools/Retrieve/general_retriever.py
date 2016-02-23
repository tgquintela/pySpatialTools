
"""
General retriever
-----------------
General retriever which acts as a container of some son class we wanna code
in order to fit with the requisits of the module and be useful for the
retriever task.

"""

import numpy as np
from retrievers import Retriever


class GeneralRetriever(Retriever):
    """General retriever. It is a general way to instantiate a retrieve with
    its needing from an object with few requisits.

    Requisits
    ---------
    retriever.data: object with different elements, __getitem__ with one index
        and element-like __eq__ function.
    retriever.retrieve_neighs
    retriever.__init__ needs to call _initialization()

    """

    _default_ret_val = None

    def retrieve_neighs(self, i_loc, info_i={}, ifdistance=None, output=0):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        """
        ## 0. Prepare variables
        info_i = self._get_info_i(i_loc, info_i)
        ifdistance = self._ifdistance if ifdistance is None else ifdistance
        ## 1. Retrieve neighs
        neighs, dists = self.retriever.retrieve_neighs(i_loc, info_i,
                                                       ifdistance)
        ## 2. Exclude auto if it is needed
        neighs_info = self._format_output(i_loc, neighs, dists)
        return neighs_info

    def _format_output(self, i_locs, neighs, dists):
        "Format output."
        neighs, dists = self._exclude_auto(i_locs, neighs, dists)
        neighs_info = self._output_map(i_locs, (neighs, dists))
        dists = self.relative_pos(self.data_output[neighs])
        return neighs_info


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
