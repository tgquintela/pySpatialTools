
"""
Density assignation process
---------------------------
Module which is oriented to compute the assignation of the density variable
to a point considering a spatial distributions of features in the space.

"""

import numpy as np
from pySpatialTools.utils.util_external.ProcessTools import Processer
from density_assignation import general_density_assignation

m = "Assignation of a quantity to a point given spatial density distribution."


###############################################################################
################################# ProcessClass ################################
###############################################################################
class DensityAssign_Process(Processer):
    """Class which assigns a density value given a spatial point distribution
    of features.
    """

    ### Class parameters
    proc_name = "Density assignation process"  # Name of the process

    def _initialization(self):
        ## Logger info
        self.lim_rows = 0  # Lim of rows done in a bunch. For matrix comp or information
        self.logfile = None  # Log file
        ## Bool options
        self.bool_inform = False  # Give information of the process
        self.bool_matrix = False  # compute matrix
        self.proc_desc = m
        ## Process descriptors
        self.time_expended = 0.  # Time expended along the process
        self.n_procs = 0  # Number of cpu used in parallelization (0 no parallel)
        self.subproc_desc = []
        self.t_expended_subproc = []
        self.proc_desc = "Computation %s with %s"

    def __init__(self, logfile, retriever, lim_rows=0, n_procs=0,
                 proc_name=""):
        """Instantiation of a density assignation process class.

        Parameters
        ----------
        logfile: str
            the log file to log all the information about that process.
        retriever: pst.Retrieve.SpaceRetriever
            the retriever object instantiation to obtain the neighbourhood.
        lim_rows: int (default=0)
            the limit rows to uninform about the loop process.
        n_procs: int (default=0)
            number of processors.
        proc_name: str (default="")
            the process name.

        """
        # Initialization
        self._initialization()
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

    def compute_density(self, locs, data, datavars, info_ret, params):
        """Compute density of the locations locs from a spatial distribution
        of features given in data.

        Parameters
        ----------
        locs: np.ndarray
            the spatial information of the retrievable 
        data: pd.DataFrame
            the information
        datavars: dict of list
        the variables of 'loc_vars' and 'feat_vars'.
        datavars
        info_ret
        params: dict
            t
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

    Parameters
    ----------
    locs: np.ndarray
        the spatial information of the retrievable 
    data: pd.DataFrame
        the information
    datavars: dict of list
        the variables of 'loc_vars' and 'feat_vars'.
    info_ret: functions, str
        function of weighs assignation. It transforms the distance to weights.
    params: dict
        parameters needed to apply f_dens.

    Returns
    -------
    dens_assignation: array_like, shape(n)
        mesasure of each location given.

    """

    ## 0. Computation of initial variables
    locs = np.array(locs)
    if type(data) == np.ndarray:
        locs_data, pop_data = data, datavars
    else:
        locs_data = np.array(data[datavars['loc_vars']])
        pop_data = np.array(data[datavars['feat_vars']])
    # Defining the retriever
    retriever = retriever(locs, ifdistance=True)

    ## 1. Computation of assignation to point
    dens_assignation = general_density_assignation(locs_data, retriever,
                                                   info_ret, pop_data,
                                                   **params)

    return dens_assignation
