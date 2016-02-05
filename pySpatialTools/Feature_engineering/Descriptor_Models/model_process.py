
"""
Model process
-------------
This module contains the main class to execute the process.

TODO
----
matrix and corr in the same process
problem with reindices and matrix
Compute in bunchs

"""

from os.path import join
import shelve
import networkx as nx

from model_utils import filter_with_random_nets

from pySpatialTools.IO import create_model_report
from pythonUtils.ProcessTools import Processer


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class SpatialDescriptorModelProcess(Processer):
    """Class which performs the spatialmodel computation. This process assigns
    a descriptors for each point in the dataset regarding their raw features
    and the spatial relation between them.

    ===============================
    Functionalities:
    - Compute net from data (parallel/sequential)
    - Compute matrix for trainning approach (parallel/sequential)

    ================================
    Problems:
    - ...
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

    def __init__(self, sp_descriptormodel, logfile, lim_rows=0, n_procs=0,
                 proc_name="Model computation"):
        # Logfile
        self.logfile = logfile
        ## Descriptor model
        self.descriptormodel = sp_descriptormodel

        # Other parameters
        self.lim_rows = lim_rows
        self.bool_inform = True if self.lim_rows != 0 else False
        self.n_procs = n_procs
        self.proc_name = proc_name
        self.proc_desc = "Computation %s with %s"

    ###########################################################################
    ######################## Measure computations #############################
    ###########################################################################
    def compute_measure_all(self):
        """Main function for building the index of the selected model. This
        function acts as swicher between the different possibilities:
        - Parallel
        - Sequential

        """

        ## 0. Setting needed variables (TODO: cambiar sptypemodel)
        m_aux0 = self.sp_descriptormodel.sptypemodel
        name_desc = self.sp_descriptormodel.name_desc
        self.proc_desc = self.proc_desc % (m_aux0, name_desc)
        t00 = self.setting_global_process()

        ## 1. Computation of the measure (parallel if)
        corr = self.compute_mea_sequ_generic()

        ## 2. Building a net
        corr = self.sp_descriptormodel.to_complete_measure(corr)
        ## Closing process
        self.close_process(t00)
        return corr

    def compute_mea_sequ_generic(self):
        """Main function to perform spatial correlation computation in a
        sequential mode.
        """
        ## 0. Intialization of needed variables
        N_t, n_calc = self.descriptormodel.model_dim
        corr = self.descriptormodel.intialization_output(n_calc)
        ## 1. Computation of local spatial correlations
        ## Begin to track the process
        t0, bun = self.setting_loop(N_t)
        for i in xrange(N_t):
            ## Loop over the possible reindices
            for k in range(n_calc):
                # 1. Retrieve local descriptors
                val_i, corr_i = self.descriptormodel.compute_net_i()
                # 2. Aggregation
                corr[val_i, :, :] =\
                    self.descriptormodel.add2result(corr[val_i, :, :], corr_i)
            ## Finish to track this process
            t0, bun = self.messaging_loop(i, t0, bun)
        return corr

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
