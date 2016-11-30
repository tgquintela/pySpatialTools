
"""
Retriever api creation
----------------------
The utils to interact easily with the retriever creation and instantiation.

"""

#
#retriever_info = locs, autolocs, pars_ret,
#retrieving_info = info_ret, autoexclude, ifdistance, info_f, output_map
#    perturbations
#    relative_pos
# inputs_configuration = input_map, bool_input_idx, constant_info,

from pySpatialTools.Retrieve import *


def set_retriver_manager(input_info):
    """The main function to set retriever.

    Parameters
    ----------
    input_info: tuple
        the information to create the retriever object.

    Returns
    -------
    ret: pySpatialTools.BaseRetriever
        the retriever instance for the parameters input.

    """
    pass


def set_retriever(input_info):
    """The main function to set retriever.

    Parameters
    ----------
    input_info: tuple
        the information to create the retriever object.

    Returns
    -------
    ret: pySpatialTools.BaseRetriever
        the retriever instance for the parameters input.

    """
    ## 0. Parsing input
    # Filtering class info
    if type(input_info[0]) == str:
        clase = eval(input_info[0])
    else:
        clase = input_info[0]
    # Filtering parameters info
    l = len(input_info)
    if type(input_info[1]) == dict:
        assert(all([type(input_info[i]) == dict for i in range(1, l)]))
        parameters = {}
        for i in range(1, l):
            parameters.update(input_info[i])
    else:
        ## TODO
        pass
    ret = clase(parameters)
    return ret
