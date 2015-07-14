
"""
Module which contains the abstract classes used in the
Supermodule Models and the process to apply model used to a particular data.
"""

from model_utils import filter_with_random_nets
from Mscthesis.IO.model_report import create_model_report
from os.path import join
import shelve

import networkx as nx
import numpy as np

import multiprocessing as mp
import time

from Mscthesis.IO.write_log import Logger

from aux_functions import init_compl_arrays


###############################################################################
########### Global variables needed for this module
###############################################################################
message0 = """========================================
Start inferring %s:
--------------------
(%s)

"""
message1 = "Processing %s:"
message2 = "completed in %f seconds.\n"
message2a = " %s bunch of %s rows completed in %f seconds.\n"
message3 = "Total time expended computing net: %f seconds.\n"
message_close = '----------------------------------------\n'

m_debug1 = "Retrieving neighs in %f seconds."
m_debug2 = "Computing M-index in %f seconds."
m_debug3 = "%f"
m_debug4 = "Computing M-index for k=%s in %f seconds."

###############################################################################
###############################################################################


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

    def __init__(self, logfile, retriever, descriptormodel, typevars,
                 lim_rows=None, n_procs=None, proc_name=None):
        # Logfile
        self.logfile = Logger(logfile)
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

    def compute_measure_all(self, df, info_ret=None, cond_agg=None,
                            reindices=None):
        """Main function for building the index of the selected model. This
        function acts as swicher between the different possibilities:
        - Parallel from data/neighs/(agg/preagg)
        - Sequential from data/neighs/(agg/preagg)
        """

        ## 0. Setting needed variables
        m_aux0 = "Training matrix" if self.bool_matrix else "Net"
        m_aux1 = "Trial0" if self.proc_name is None else self.proc_name
        self.logfile.write_log(message0 % (m_aux0, m_aux1))
        t00 = time.time()
        # Preparing needed vars
        aux = init_compl_arrays(df, self.typevars, info_ret, cond_agg)
        locs, feat_arr, info_ret, cond_agg = aux
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
        t_expended = time.time()-t00
        self.logfile.write_log(message3 % t_expended)
        self.logfile.write_log(message_close)
        self.time_expended = t_expended

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
            corr_loc = []
        else:
            n_vals0, n_vals1 = self.descriptormodel.model_dim
            corr_loc = np.zeros((n_vals0, n_vals1, n_calc))
        ## Begin to track the process
        t0 = time.time()
        bun = 0
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
                    corr_loc.append(corr_loc_i)
                else:
                    corr_loc[val_i, :, k] += corr_loc_i
            ## Finish to track this process
            if self.bool_inform and (i % self.lim_rows) == 0 and i != 0:
                t_sp = time.time()-t0
                bun += 1
                self.logfile.write_log(message2a % (bun, self.lim_rows, t_sp))
                t0 = time.time()
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


###########################################################################
############################ Auxiliar classes #############################
###########################################################################
class DescriptorModel:
    "General class for descriptor models."
    model_dim = (0, 0)  # check compute_descriptors

    def get_characterizers(self, i, k, feat_arr, point_i, reindices,
                           retriever, info_ret, cond_agg):
        """Retrieve local characterizers for i element and k permutation. It
        returns the column index in the output matrix correlation (val_i) and
        trivial descriptors of the neighbourhood (vals). These values are used
        for the specific model function compute_descriptors.
        """
        # Retrieve neighs
        info_i, cond_i = info_ret[i], cond_agg[i]
        neighs, type_n = retriever.retrieve_neighs(point_i, cond_i, info_i)
        # Get vals
        val_i = self.compute_value_i(i, k, feat_arr, reindices)
        vals = self.compute_vals_nei(retriever.aggfeatures, feat_arr, neighs,
                                     reindices, k, type_n)
        # Get characterizers
        characs = self.integrate_vals(vals, type_n)

        return val_i, characs
