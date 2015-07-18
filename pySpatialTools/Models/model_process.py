
"""
Model process
-------------
This module contains the main class to execute the process.

TODO
----
matrix with np.array not appending lists
Class a a process Class

"""

from os.path import join
import shelve
import networkx as nx
import numpy as np
import multiprocessing as mp
import time

from model_utils import filter_with_random_nets
from Mscthesis.IO.model_report import create_model_report
from aux_functions import init_compl_arrays


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class ModelProcess():
    """Abstract class for performs the process of computation of the models.
    ===============================
    Functionalities:
    - Compute net from data (parallel/sequential)
    - Compute net from precomputed neighs (parallel/sequential)
    - Compute net from agg points (parallel/sequential)
    - Compute matrix for trainning approach (parallel/sequential)

    ================================
    Problems:
    - Mask neighs
    - Get and compute descriptors (online/file)
    - Aggregate descriptors (measure dependant)

    ================================
    TODO:
    - lim_rows matrix computation (auxiliary folder to save)
    - ...

    """

    ### Class parameters
    ## Process descriptors
    time_expended = 0.  # Time expended along the process
    n_procs = 0  # Number of cpu used in parallelization (0 no parallel)
    proc_name = None  # Name of the process
    ## Logger info
    lim_rows = 0  # Lim of rows done in a bunch. For matrix comp or information
    logfile = None  # Log file
    ## Bool options
    bool_inform = False  # Give information of the process
    bool_matrix = False  # compute matrix

    subproc_desc = []
    t_expended_subproc = []

    def __init__(self, logfile, retriever, descriptormodel, typevars,
                 lim_rows=0, n_procs=0, proc_name=""):
        # Logfile
        self.logfile = logfile
        ## Retriever
        self.retriever = retriever
        ## Descriptor model
        self.descriptormodel = descriptormodel
        ## Type of variables
        self.typevars = typevars  # filter typevars

        # Other paramters
        self.lim_rows = lim_rows
        self.n_procs = n_procs
        self.proc_name = proc_name
        self.proc_desc = proc_desc

    ###########################################################################
    ######################## Measure computations #############################
    ###########################################################################
    def compute_net(self, df, info_ret=None, cond_agg=None, reindices=None):
        """Main function for the computation of the matrix. It acts as a
        wrapper over the compute_measure_all function.
        """
        self.bool_matrix = False
        net = self.compute_measure_all(df, info_ret, cond_agg, reindices)
        return net

    def compute_matrix(self, df, info_ret=None, cond_agg=None, reindices=None):
        """Main function for the computation of the matrix. It acts as a
        wrapper over the compute_measure_all function.
        """
        self.bool_matrix = True
        matrix = self.compute_measure_all(df, info_ret, cond_agg, reindices)
        return matrix

    def compute_measure_all(self, df, reindices=None):
        """Main function for building the index of the selected model. This
        function acts as swicher between the different possibilities:
        - Parallel from data/neighs/(agg/preagg)
        - Sequential from data/neighs/(agg/preagg)
        """

        ## 0. Setting needed variables
        t00 = self.setting_global_process()
        m_aux0 = "Training matrix" if self.bool_matrix else "Net"
        m_aux1 = "Trial0" if self.proc_name is None else self.proc_name
        self.logfile.write_log(message0 % (m_aux0, m_aux1))


        # Preparing needed vars
        aux = init_compl_arrays(df, self.typevars, reindices)
        locs, feat_arr, info_ret, cond_agg, reindices = aux
        N_t = df.shape[0]
        # clean unnecessary
        del df
        # Bool options
        self.bool_inform = True if self.lim_rows is not None else False

        ## 1. Computation of the measure (parallel if)
        corr_loc = self.compute_mea_sequ_generic(locs, feat_arr, info_ret,
                                                 cond_agg, reindices)

        ## 2. Building a net
        corr_loc = self.descriptormodel.to_complete_measure(corr_loc, N_t)
        ## Closing process
        self.close_process(t00)
        return corr_loc, type_vals, N_x

    def compute_mea_sequ_generic(self, locs, feat_arr, info_ret, cond_agg,
                                 reindices):
        """Main function to perform spatial correlation computation in a
        sequential mode using aggregated information given by a '''''file'''''.
        """

        ## 0. Intialization of needed variables
        N_t = reindices.shape[0]
        n_calc = reindices.shape[1]

        ## 1. Computation of local spatial correlations
        if self.bool_matrix:
            _, n_vals1 = self.descriptormodel.model_dim
            corr_loc = np.zeros((N_t, n_vals1))
        else:
            n_vals0, n_vals1 = self.descriptormodel.model_dim
            corr_loc = np.zeros((n_vals0, n_vals1, n_calc))
        ## Begin to track the process
        t0, bun = time.time(), 0
        for i in xrange(N_t):
            ## Obtaining neighs of a given point
            point_i = locs[i, :].reshape(1, locs.shape[1])
            ## Loop over the possible reindices
            for k in range(n_calc):
                # 1. Retrieve local characterizers
                val_i, chars =\
                    self.descriptormodel.get_characterizers(i, k, feat_arr,
                                                            point_i, reindices,
                                                            self.retriever,
                                                            info_ret, cond_agg)
                # 2. Computation of the local measure
                corr_loc_i =\
                    self.descriptormodel.compute_descriptors(chars, val_i)
                # 3. Aggregation
                if self.bool_matrix:
                    corr_loc[i, :] = corr_loc_i
                else:
                    corr_loc[val_i, :, k] += corr_loc_i
            ## Finish to track this process
            t0, bun = self.messaging_loop(i, t0, bun)
        return corr_loc

    ###########################################################################
    ######################### Statistic significance ##########################
    ###########################################################################
    def filter_with_random_nets(self, nets, p_thr):
        "Filter non-significant weiths."
        net, random_nets = nets[:, :, 0], nets[:, :, 1:]
        net = filter_with_random_nets(net, random_nets, p_thr)
        return net

    ###########################################################################
    ############################# Outputs #####################################
    ###########################################################################
    def to_report(self, net, sectors, dirname, reportname):
        "Generate a folder in which save the report data exported."
        fig1, fig2 = create_model_report(net, sectors, dirname, reportname)
        return fig1, fig2

    def to_pajek(self, net, sectors, netfiledata, filenamenet):
        "Export net to pajek format *.net"
        net_out = nx.from_numpy_matrix(net)
        n_sects = len(sectors)
        net_out = nx.relabel_nodes(net_out, dict(zip(range(n_sects), sectors)))
        nx.write_pajek(net_out, join(netfiledata, filenamenet))

    def save_net_to_file(self, net, sectors, N_t, N_x, outputfile):
        "Save interesting quantities in a external file."
        database = shelve.open(outputfile)
        database['net'] = net
        database['sectors'] = sectors
        database['time'] = self.time_expended
        database['N_t'] = N_t
        database['N_x'] = N_x
        database.close()
