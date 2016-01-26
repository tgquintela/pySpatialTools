
"""
CollectionRetrievers
--------------------
Wrapper of retrievers with a customed api oriented to interact properly with
spatial descriptor models.

"""


class CollectionRetrievers:
    """Retreiver of elements given other elements and only considering the
    information of the non-retrivable elements.
    There are different retriavables objects and different inputs.
    """

    typeret = 'collection'

    ## Elements information
    retrievers = []
    n_inputs = 0
    _map_retriever = lambda s, typeret_i: (0, 0)

    def __init__(self, retrievers, map_retriever=None):
        self._format_retrievers(retrievers)
        self._format_map_retriever(map_retriever)

    def _format_map_retriever(self, map_retriever):
        if not map_retriever is None:
            self._map_retriever = map_retriever

    def retrieve_neighs(self, i, info_i=None, ifdistance=None,
                        typeret_i=None):
        typeret_i, out_ret = self._get_type_ret(typeret_i, i)
        neighs_info = self.retrievers[typeret_i].retrieve_neighs(i, info_i,
                                                                 ifdistance,
                                                                 out_ret)
        return neighs_info

    def _get_type_ret(self, typeret_i, i):
        if typeret_i is None:
            typeret_i, out_ret = self._map_retriever(i)
        else:
            typeret_i, out_ret = typeret_i
        return typeret_i, out_ret

    def _format_retrievers(self, retrievers):
        if type(retrievers) == list:
            self.retrievers = retrievers
        else:
            raise TypeError("Incorrect type. Not retrievers list.")
        ## WARNING: By default it is determined by the first retriever
        self.n_inputs = len(self.retrievers[0].data_input)
