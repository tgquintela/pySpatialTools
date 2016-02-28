
"""
General retriever
-----------------
General retriever which acts as a container of some son class we wanna code
in order to fit with the requisits of the module and be useful for the
retriever task.

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
    retriever.__init__ needs to call _initialization()

    """

    _default_ret_val = None

    def _format_output_exclude(self, i_locs, neighs, dists, output=0):
        "Format output."
        neighs, dists = self._exclude_auto(i_locs, neighs, dists)
        neighs_info = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs_info

    def _format_output_noexclude(self, i_locs, neighs, dists, output=0):
        "Format output."
        neighs_info = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs_info
