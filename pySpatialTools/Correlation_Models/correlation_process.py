
"""
Correlation process
-------------------
The computation of the correlation matrices process from a matrix of features
description of the data.
"""

import numpy as np


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class CorrModelProcess(Processer):
    """Class which performs the spatial correlation model computation. This
    process uses the descriptors of the data to comopute spatial correlation of
    descriptors.

    assigns a descriptors for each point in the dataset regarding their raw features
    and the spatial relation between them.

    ===============================
    Functionalities:
    - Compute net from data (parallel/sequential)

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

    def __init__(self, logfile, retriever, corrmodel, typevars,
                 lim_rows=0, n_procs=0, proc_name="Model computation"):
        # Logfile
        self.logfile = logfile
        ## Retriever
        self.retriever = retriever
        ## Descriptor model
        self.corrmodel = corrmodel
        ## Type of variables
        self.typevars = typevars  # filter typevars

        # Other parameters
        self.lim_rows = lim_rows
        self.bool_inform = True if self.lim_rows != 0 else False
        self.n_procs = n_procs
        self.proc_name = proc_name
        self.proc_desc = "Computation %s with %s"

    ###########################################################################
    ######################## Measure computations #############################
    ###########################################################################
    def compute_net(self, df, reindices=None):
        """Main function for the computation of the matrix. It acts as a
        wrapper over the compute_measure_all function.
        """
        self.bool_matrix = False
        net = self.compute_measure_all(df, reindices)
        return net

    def compute_matrix(self, df, reindices=None):
        """Main function for the computation of the matrix. It acts as a
        wrapper over the compute_measure_all function.
        """
        self.bool_matrix = True
        matrix = self.compute_measure_all(df, reindices)
        return matrix

    def compute_measure_all(self, df, reindices=None):
        """Main function for building the index of the selected model. This
        function acts as swicher between the different possibilities:
        - Parallel from data/neighs/(agg/preagg)
        - Sequential from data/neighs/(agg/preagg)
        """

        ## 0. Setting needed variables
        m_aux0 = "correlation matrix"
        name_desc = self.descriptormodel.name_desc
        self.proc_desc = self.proc_desc % (m_aux0, name_desc)
        t00 = self.setting_global_process()

        # Preparing needed vars
        aux = init_compl_arrays(df, self.typevars, reindices)
        locs, feat_arr, info_ret, cond_agg, reindices = aux
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

    def compute_mea_sequ_generic(self, locs, feat_arr, info_ret, cond_agg,
                                 reindices):
        """Main function to perform spatial correlation computation in a
        sequential mode using aggregated information given by a '''''file'''''.
        """

        ## 0. Intialization of needed variables
        N_t = reindices.shape[0]

        ## 1. Computation of local spatial correlations
        n_vals0, n_vals1 = self.corrmodel.model_dim
        n_calc = reindices.shape[1]
        corr_loc = np.zeros((n_vals0, n_vals1, n_calc))

        ## Begin to track the process
        t0, bun = self.setting_loop(N_t)
        for i in xrange(N_t):
            ## Obtaining neighs of a given point
            point_i = locs[i, :].reshape(1, locs.shape[1])
            ## Loop over the possible reindices
            for k in range(n_calc):
                # 1. Retrieve local characterizers
                val_i, chars =\
                    self.corrmodel.get_characterizers(i, k, feat_arr, point_i,
                                                      reindices,
                                                      self.retriever, info_ret,
                                                      cond_agg)
                # 2. Computation of the local measure
                corr_loc_i =\
                    self.corrmodel.compute_descriptors(chars, val_i)
                # 3. Aggregation
                corr_loc[val_i, :, k] += corr_loc_i
            ## Finish to track this process
            t0, bun = self.messaging_loop(i, t0, bun)
        return corr_loc
