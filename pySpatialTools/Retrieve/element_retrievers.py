
"""
Element_retrievers
------------------
The module which contains a retriever of generic elements.

"""

###############################################################################
############################# Element Retrievers ##############################
###############################################################################
class ElementRetriever(Retriever):
    """Retreiver of elements given other elements and only considering the
    information of the non-retrivable elements.
    """
    typeret = 'element'

    def __init__(self, locs, info_ret=None, autolocs=None, pars_ret=None,
                 flag_auto=True, ifdistance=False, info_f=None, tags=None):
        "Creation a point retriever class method."
         # Retrieve information
        pars_ret = self.format_pars_ret(pars_ret)
        self.retriever = define_kdretriever(locs, **pars_ret)
        ## Info_ret mangement
        if type(info_ret).__name__ == 'function':
            self.info_f = info_ret
        else:
            self.info_f = info_f
        self.info_ret = self.default_ret_val if info_ret is None else info_ret
        self.ifdistance = ifdistance
        # Location information
        self.autolocs = True if autolocs is None else False
        self.locs = None if autolocs is None else autolocs
        # Filter information
        self.flag_auto = flag_auto

    def discretize(self, i_locs):
        """Format the index retrieving for the proper index of retrieving of
        the type of retrieving.
        """
        ###### TODO: Correct that
        if self.check_coord:
            return -1 * np.ones(i_locs.shape[0])
        return i_locs

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def format_pars_ret(self, pars_ret):
        "Format the parameters of retrieval."
        if pars_ret is not None:
            pars_ret = int(pars_ret)
        pars_ret = {'leafsize': pars_ret}
        return pars_ret

    def format_output(self, i_locs, neighs, dists):
        "Format output."
        neighs, dists = self.exclude_auto(i_locs, neighs, dists)
        return neighs, dists


def define_kdretriever(locs, leafsize=None):
    "Define a kdtree for retrieving neighbours."
    leafsize = locs.shape[0]
    leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
    return KDTree(locs, leaf_size=leafsize)
