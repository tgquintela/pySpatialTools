
"""
pipeline
--------
pipeline module groups the utilities and the pipeline_process class in order
to compute some functionalities we want by only adding the parameters and
functions to the pipeline class.
"""

import numpy as np


def creation_general_process():
    """"""
    pass


def _process_discretizor(discretizor):
    """Prepare discretizor to use."""
    if type(discretizor) == tuple:
        locs, regs = discretizor
        assert(len(locs) == len(regs))
    else:
        pass
    return locs, regs


def _process_parameters(parameters_ret):
    assert(type(parameters_ret) == dict)
    parameters_ret['ifdistance': True]
    return parameters_ret


def create_m_out_inv_disc(discretization):
    if type(discretization) == np.ndarray:
        assert(len(discretization) == len(discretization.ravel()))
        discretization = discretization.ravel()
        u_regs = np.unique(discretization)

        def m_out(x):
            neighs = []
            for x_i in x:
                neighs_i = []
                if x not in u_regs:
                    pass
    else:
        assert(type(discretization) == tuple)
        assert(len(discretization) == 2)
        regs = discretization[0].discretize(discretization[1])
        m_out = create_m_out_inv_disc(regs)
    return m_out


###############################################################################
######################## Main creation spdesc functions #######################
###############################################################################
def _discretization_parsing_creation(discretization_info):
    pass


def _retrieve_parsing_creation(retrieve_info):
    pass
