
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

import numpy as np
from tools_retriever import create_aggretriever
from pySpatialTools.utils.util_classes import Spatial_RetrieverSelector,\
    format_selection
from pySpatialTools.utils.util_classes.neighs_info import join_by_iss

inttypes = [int, np.int32, np.int64]


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
        self.staticneighs = True
        ## TODO:
        self.selector = (0, 0)

    def __init__(self, retrievers, selector_retriever=None):
        self._initialization()
        self._format_retrievers(retrievers)
        self._format_selector(selector_retriever)

    def __len__(self):
        return len(self.retrievers)

    def __getitem__(self, i_ret):
        if i_ret < 0 or i_ret >= len(self.retrievers):
            raise IndexError("Not correct index for features.")
        return self.retrievers[i_ret]

    def __iter__(self):
        """It assumes preferent retriever 0."""
        ## If constant selected retriever
        for neighs_info in self[0]:
            yield neighs_info
        ## If mapper active (TODO)

    def _retrieve_neighs_constant(self, i, typeret_i=None):
        """Retrieve neighbourhood under conditions of ifdistance or others
        interior parameters. Constant typeret_i assumption."""
        typeret_i, out_ret = self.get_type_ret(i, typeret_i)
        neighs_info =\
            self.retrievers[typeret_i].retrieve_neighs(i, output=out_ret)
        return neighs_info

    def _retrieve_neighs_variable(self, i, typeret_i=None):
        """Retrieve neighbourhood under conditions of ifdistance or others
        interior parameters. Variable typeret_i assumption."""
        typeret_i, out_ret = self.get_type_ret(i, typeret_i)
        i = [i] if type(i) in inttypes else i
        typeret_i = [typeret_i] if type(typeret_i) != list else typeret_i
        out_ret = [out_ret] if type(out_ret) != list else out_ret
        neighs = []
        for j in range(len(typeret_i)):
            neighs_info = self.retrievers[typeret_i[j]].\
                retrieve_neighs(i[j], output=out_ret[j])
            neighs.append(neighs_info)
        neighs_info = join_by_iss(neighs)
        return neighs_info

    def _retrieve_neighs_general(self, i, typeret_i=None):
        """Retrieve neighbourhood under conditions of ifdistance or others
        interior parameters. No typeret_i assumption."""
        typeret_i, out_ret = self.get_type_ret(i, typeret_i)
        if type(typeret_i) == list:
            i = [i] if type(i) in inttypes else i
            neighs = []
            for j in range(len(typeret_i)):
                neighs_info = self.retrievers[typeret_i[j]].\
                    retrieve_neighs(i[j], output=out_ret[j])
                neighs.append(neighs_info)
            neighs_info = join_by_iss(neighs)
        else:
            neighs_info =\
                self.retrievers[typeret_i].retrieve_neighs(i, output=out_ret)
        return neighs_info

    def compute_nets(self, kret=None):
        """Compute all the possible relations if there is a common
        (homogeneous) ouput.
        """
        ## Check that match conditions (TODO)
        ## Format kret
        kret = range(len(self.retrievers)) if kret is None else kret
        kret = [kret] if type(kret) == int else kret
        ## Compute
        nets = []
        for r in kret:
            nets.append(self.retrievers[r].compute_neighnets(self.selector))
        return nets

    ######################### Auxiliar administrative #########################
    ###########################################################################
    def add_retrievers(self, retrievers):
        """Add new retrievers."""
        self._format_retrievers(retrievers)

    def set_neighs_info(self, bool_input_idx):
        """Setting the neighs info of the retrievers."""
        for i in range(len(self)):
            self.retrievers[i]._format_neighs_info(bool_input_idx)

    def set_selector(self, selector):
        """Set a common selector in order to not depend on continous external
        orders.
        """
        self._format_selector(selector)

    ################################ Formatters ###############################
    ###########################################################################
    def _format_retrievers(self, retrievers):
        if type(retrievers) == list:
            self.retrievers += retrievers
        elif retrievers.__name__ == 'pySpatialTools.Retriever':
            self.retrievers.append(retrievers)
        elif not(type(retrievers) == list):
            raise TypeError("Incorrect type. Not retrievers list.")
        ## WARNING: By default it is determined by the first retriever
        ret_n_inputs = [len(self.retrievers[i]) for i in range(len(self))]
        assert(all([len(self.retrievers[0]) == r for r in ret_n_inputs]))
        assert(all([self.retrievers[0].k_perturb == r.k_perturb
                    for r in self.retrievers]))
        self.k_perturb = self.retrievers[0].k_perturb
        self.n_inputs = len(self.retrievers[0])
        self._format_staticneighs()

    def _format_staticneighs(self):
        """Format staticneighs."""
        ## Set staticneighs
        n_ret = len(self.retrievers)
        self.staticneighs = self[0].staticneighs
        aux = [self[i].staticneighs == self.staticneighs for i in range(n_ret)]
        assert(aux)

    def _format_selector(self, selector):
        """Programable get_type_ret."""
        if selector is None:
            self.get_type_ret = self._general_get_type_ret
            self.retrieve_neighs = self._retrieve_neighs_general
        elif type(selector) == tuple:
            self.selector = selector
            self.get_type_ret = self._static_get_type_ret
            self.retrieve_neighs = self._retrieve_neighs_constant
        else:
            self.selector = Spatial_RetrieverSelector(selector)
            self.get_type_ret = self._selector_get_type_ret
            self.retrieve_neighs = self._retrieve_neighs_variable
            self.selector.assert_correctness(self)

    ################################# Type_ret ################################
    ###########################################################################
    ## Formatting the selection of path from i information.
    ##
    ## See also:
    ## ---------
    ## pst.FeaturesManager
    #########################
    def _general_get_type_ret(self, i, typeret_i=None):
        """Interaction with the selector. Using upside information of selection
        or the own selector the manager owns."""
        if typeret_i is None:
            typeret_i, out_ret = self.selector
        else:
            if type(i) == list:
                typeret_input = typeret_i
                ## TOMOVE to selectors
                typeret_i, out_ret = [], []
                for j in range(len(i)):
                    typeret_i.append(typeret_input[j][0])
                    out_ret.append(typeret_input[j][1])
            else:
                typeret_i, out_ret = typeret_i
        return typeret_i, out_ret

    def _static_get_type_ret(self, i, typeret_i=None):
        """Interaction with the selector. Using upside information of selection
        or the own selector the manager owns."""
        typeret_i, out_ret = self.selector
        return typeret_i, out_ret

    def _selector_get_type_ret(self, i, typeret_i=None):
        """Get information only from selector."""
        selection = format_selection(self.selector[i])
        typeret_i, out_ret = selection
        return typeret_i, out_ret

    ######################## Perturbation management ##########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Adding perturbations to retrievers."""
        for i_ret in range(len(self.retrievers)):
            self.retrievers[i_ret].add_perturbations(perturbations)
        ## Update staticneighs
        self._format_staticneighs()

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
