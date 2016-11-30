
"""
General retriever
-----------------
General retriever which acts as a container of some son class we wanna code
in order to fit with the requisits of the module and be useful for the
retriever task.

"""

from retrievers import BaseRetriever


class GeneralRetriever(BaseRetriever):
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
        """Format output without excluding the same i.

        Parameters
        ----------
        i_loc: int, list or np.ndarray or other
            the information of the element (as index or the whole spatial
            information of the element to retieve its neighborhood)
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss
        output: int (default = 0)
            the number of output mapper function selected.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss

        """
        neighs, dists = self._exclude_auto(i_locs, neighs, dists)
        neighs_info = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs_info

    def _format_output_noexclude(self, i_locs, neighs, dists, output=0):
        """Format output without excluding the same i.

        Parameters
        ----------
        i_loc: int, list or np.ndarray or other
            the information of the element (as index or the whole spatial
            information of the element to retieve its neighborhood)
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss
        output: int (default = 0)
            the number of output mapper function selected.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss

        """
        neighs_info = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs_info
