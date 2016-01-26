
"""
General retriever
-----------------

"""

from retrievers import Retriever


class GeneralRetriever(Retriever):
    """General retriever. It is a general way to instantiate a retrieve with
    its needing from an object with few requisits.

    Requisits
    ---------
    retriever.data: object with different elements, __getitem__ with one index
        and element-like __eq__ function.
    retriever.retrieve_neighs


    """

    _default_ret_val = None
    _heterogenous_output = False
    _heterogenous_input = False

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
