
"""
DescriptorModel process
-----------------------
This module contains the main class to execute the process to compute the
descriptors from a spatial density of features dataset.


TODO
----
Use interpolator and descriptor models to compute regions distance.

"""

from os.path import join
import shelve
import networkx as nx
import numpy as np
import multiprocessing as mp

from model_utils import filter_with_random_nets
from pySpatialTools.utils.transformation_utils import split_df,\
    compute_reindices
from pySpatialTools.IO import create_model_report
from pythonUtils.ProcessTools import Processer


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class DescriptorModelProcess(Processer):
    """Class which performs the spatialmodel computation. This process assigns
    a descriptors for each point in the dataset regarding their raw features
    and the spatial relation between them.

    ===============================
    Functionalities:
    - Compute matrix for trainning approach (parallel/sequential)

    ================================
    Problems:

    ================================
    TODO:

    """

    ### Class parameters
    ## Process descriptors
    time_expended = 0.  # Time expended along the process
    n_procs = 0  # Number of cpu used in parallelization (0 no parallel)
    proc_name = ""  # Name of the process
    proc_desc = ""
    ## Logger info
    lim_rows = 0  # Lim of rows done in a bunch. For matrix comp or information
    logfile = None  # Log file
    ## Bool options
    bool_inform = False  # Give information of the process
    bool_matrix = False  # compute matrix

    subproc_desc = []
    t_expended_subproc = []

    def __init__(self, logfile, retriever, descriptormodel, typevars,
                 lim_rows=0, n_procs=0, proc_name="Descriptors computation"):
        # Logfile
        self.logfile = logfile
        ## Descriptor model
        self.descriptormodel = descriptormodel
        ## Type of variables
        self.typevars = typevars  # filter typevars

        # Other parameters
        self.lim_rows = lim_rows
        self.bool_inform = True if self.lim_rows != 0 else False
        self.n_procs = n_procs
        self.proc_name = proc_name
        self.proc_desc = "Computation %s with %s"

    def compute_descriptors(self, df, reindices=None):
        """Main function for building the index of the selected model. This
        function acts as swicher between the different possibilities:
        - Parallel from data/neighs/(agg/preagg)
        - Sequential from data/neighs/(agg/preagg)
        """

        ## 0. Setting needed variables
        m_aux0 = "training matrix" if self.bool_matrix else "net"
        name_desc = self.descriptormodel.name_desc
        self.proc_desc = self.proc_desc % (m_aux0, name_desc)
        t00 = self.setting_global_process()

        # Preparing needed vars
        locs, feat_arr, info_ret, cond_agg = split_df(df, self.typevars)
        reindices = compute_reindices(df, reindices)
        N_t = df.shape[0]
        # clean unnecessary
        del df

        ## 1. Computation of the measure (parallel if)
        corr_loc = self.compute_mea_sequ_generic(locs, feat_arr, info_ret,
                                                 cond_agg, reindices)

        ## 2. Building a net
        corr_loc = self.descriptormodel.to_complete_measure(corr_loc, N_t)
        ## Closing process
        self.close_process(t00)
        return corr_loc
