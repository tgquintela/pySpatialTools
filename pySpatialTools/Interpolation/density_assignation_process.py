
"""
Density assignation process
---------------------------
Module which is oriented to compute the assignation of the density variable
to a point considering a spatial distributions of features in the space.

"""

import numpy as np
from pythonUtils.ProcessTools import Processer
from density_assignation import general_density_assignation


###############################################################################
################################# ProcessClass ################################
###############################################################################
class DensityAssign_Process(Processer):
    """Class which assigns a density value given a spatial point distribution
    of features.
    """

    ### Class parameters
    ## Process descriptors
    time_expended = 0.  # Time expended along the process
    n_procs = 0  # Number of cpu used in parallelization (0 no parallel)
    proc_name = "Density assignation process"  # Name of the process
    proc_desc = """Assignation of a quantity to a point given spatial density
    distribution."""
    ## Logger info
    lim_rows = 0  # Lim of rows done in a bunch. For matrix comp or information
    logfile = None  # Log file
    ## Bool options
    bool_inform = False  # Give information of the process
    bool_matrix = False  # compute matrix

    subproc_desc = []
    t_expended_subproc = []

    def __init__(self, logfile, retriever, lim_rows=0, n_procs=0,
                 proc_name=""):
        "Instantiation of a density assignation process class."

        # Logfile
        self.logfile = logfile
        ## Retriever
        self.retriever = retriever

        # Other parameters
        self.lim_rows = lim_rows
        self.bool_inform = True if self.lim_rows != 0 else False
        self.n_procs = n_procs
        if proc_name != "":
            self.proc_name = proc_name
        self.proc_desc = "Computation %s with %s"

    def compute_density(self, locs, data, datavars, info_ret, params):
        """Compute density of the locations locs from a spatial distribution
        of features given in data.
        """

        d = compute_population_data(locs, data, datavars, self.retriever,
                                    info_ret, params)
        return d


###############################################################################
############################## Auxiliary functions ############################
###############################################################################
def compute_population_data(locs, data, datavars, retriever, info_ret, params):
    """Function to compute the correspondant density data to each point in locs
    given the spatial distribution of features given in data.
    """

    ## 0. Computation of initial variables
    locs = np.array(locs)

    locs_data = np.array(data[datavars['loc_vars']])
    pop_data = np.array(data[datavars['feat_vars']])

    # Defining the retriever
    retriever = retriever(locs_data)

    ## 1. Computation of assignation to point
    dens_assignation = general_density_assignation(locs, retriever, info_ret,
                                                   pop_data, **params)

    return dens_assignation
