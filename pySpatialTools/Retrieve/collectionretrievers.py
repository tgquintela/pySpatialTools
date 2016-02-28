
"""
CollectionRetrievers
--------------------
Wrapper of retrievers with a customed api oriented to interact properly with
spatial descriptor models.
This module contains the main class to mamage all retrievers without entering
in low-level programming.
As the FeaturesRetriever there are a requisit to have all the retrievers under
this manager, but in this case it is not the output, it is the input.

"""

from tools_retriever import create_aggretriever


class RetrieverManager:
    """Retreiver of elements given other elements and only considering the
    information of the non-retrivable elements.
    There are different retriavables objects and different inputs.

    See also
    --------
    FeaturesManagement.FeaturesManager

    """
    __name__ = 'pySpatialTools.RetrieverManager'
    typeret = 'manager'

    def _initialization(self):
        """Mutable globals reset."""
        ## Elements information
        self.k_perturb = 0
        self.retrievers = []
        self.n_inputs = 0
        self._selector_retriever = lambda s, typeret_i: (0, 0)
        self.staticneighs = True

    def __init__(self, retrievers, selector_retriever=None):
        self._initialization()
        self._format_retrievers(retrievers)
        self._format_map_retriever(selector_retriever)

    def __len__(self):
        return len(self.retrievers)

    def __getitem__(self, i_ret):
        if i_ret < 0 or i_ret >= len(self.retrievers):
            raise IndexError("Not correct index for features.")
        return self.retrievers[i_ret]

    def retrieve_neighs(self, i, info_i=None, ifdistance=None,
                        typeret_i=None, k=None):
        typeret_i, out_ret = self._get_type_ret(typeret_i, i, k)
        neighs_info = self.retrievers[typeret_i].retrieve_neighs(i, info_i,
                                                                 ifdistance,
                                                                 k, out_ret)
        return neighs_info

    def compute_nets(self):
        """Compute all the possible relations if there is a common
        (homogeneous) ouput.
        """
        pass

    def _get_type_ret(self, typeret_i, i, k=0):
        k = 0 if k is None else k
        if typeret_i is None:
            typeret_i, out_ret = self._selector_retriever(i, k)
        else:
            typeret_i, out_ret = typeret_i
        return typeret_i, out_ret

    def add_retrievers(self, retrievers):
        """Add new retrievers."""
        self._format_retrievers(retrievers)

    ################################ Formatters ###############################
    ###########################################################################
    def _format_map_retriever(self, selector_retriever):
        if not selector_retriever is None:
            self._selector_retriever = selector_retriever

    def _format_retrievers(self, retrievers):
        if type(retrievers) == list:
            self.retrievers = retrievers
        elif retrievers.__name__ == 'pySpatialTools.Retriever':
            self.retrievers.append(retrievers)
        elif not(type(retrievers) == list):
            raise TypeError("Incorrect type. Not retrievers list.")
        ## WARNING: By default it is determined by the first retriever
        self.n_inputs = len(self.retrievers[0])
        ## Set staticneighs
        n_ret = len(self.retrievers)
        self.staticneighs = all([self[i].staticneighs for i in range(n_ret)])

    ######################## Perturbation management ##########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Adding perturbations to retrievers."""
        for i_ret in range(len(self.retrievers)):
            self.retrievers[i_ret].add_perturbations(perturbations)
        ## Update staticneighs
        n_ret = len(self.retrievers)
        self.staticneighs = all([self[i].staticneighs for i in range(n_ret)])

    ######################### Aggregation management ##########################
    ###########################################################################
    def add_aggregations(self, discretization, regmetric=None, retriever=None,
                         pars_retriever={}, kfeat=0):
        """Add aggregations to retrievers. Only it is useful this function if
        there is only one retriever previously and we are aggregating the first
        one.
        """
        retriever = create_aggretriever(discretization, regmetric, retriever,
                                        pars_retriever)
        self.retrievers.append(retriever)
