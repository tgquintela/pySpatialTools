
"""
Auxiliar discretization parsing
-------------------------------
Parsing of information for discretization.

"""

import numpy as np
from spatialdiscretizer import SpatialDiscretizor


################################ Discretization ###############################
def _discretization_parsing_creation(discretization_info, retriever_o=None):
    """Function which uniforms the discretization info to be useful in other
    parts of the code.

    Standarts
    * (discretizator, locs)
    * (locs, regions)
    * disc
    * locs, regs, disc
    """
#    if aux is not None:
#        assert(isinstance(aux, SpatialDiscretizor))
#        assert(type(retriever_o) == np.ndarray)
#        assert(len(discretization_info) == len(retriever_o))
#        return retriever_o, aux, discretization_info
    if retriever_o is not None:
        discretization_info =\
            _discretization_information_creation(discretization_info,
                                                 retriever_o)
    assert(type(discretization_info) == tuple)
    if isinstance(discretization_info[0], SpatialDiscretizor):
        regs = discretization_info[0].discretize(discretization_info[1])
        locs = discretization_info[1]
        disc = discretization_info[0]
    else:
        assert(type(discretization_info[1]) == np.ndarray)
        assert(len(discretization_info[0]) == len(discretization_info[1]))
        if len(discretization_info) == 2:
            locs, regs = discretization_info
            disc = None
        else:
            assert(len(discretization_info) == 3)
            locs, regs, disc = discretization_info
    return locs, regs, disc


def _discretization_information_creation(discretization_o, retriever_o):
    """Function to create the information creation object."""
    if retriever_o is None:
        return discretization_o
    assert(isinstance(discretization_o, SpatialDiscretizor))
    return discretization_o, retriever_o.data_output


def _discretization_regionlocs_parsing_creation(discretization_info,
                                                elements=None,
                                                activated=False):
    """Function which uniforms the discretization info to be useful in other
    parts of the code.

    Standarts
    * discretizator
    * (discretizator, locs)
    * (discretizator, locs, regs)
    """
    if type(discretization_info) == tuple:
        assert(isinstance(discretization_info[0], SpatialDiscretizor))
        regionlocs = discretization_info[0].regionlocs
        regions = discretization_info[0].regions_id
        boolean = False
        if activated:
            regs = discretization_info[0].discretize(discretization_info[1])
            u_regs = np.unique(regs)
            boolean = True
        else:
            if elements is not None:
                u_regs = np.unique(elements)
                boolean = True
        ## Logi construction
        if boolean:
            logi = np.zeros(len(regionlocs)).astype(bool)
            for i in range(len(u_regs)):
                logi = np.logical_or(logi, regions == u_regs[i])
            regions = regions[logi]
            regionlocs = regionlocs[logi]
    else:
        assert(isinstance(discretization_info, SpatialDiscretizor))
        regionlocs = discretization_info.regionlocs
        regions = discretization_info.regions_id
    return regionlocs, regions
