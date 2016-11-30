
"""
Process spatial descriptormodel
-------------------------------
Module which grops the processer class to compute a spatial descriptor model.

"""

import copy
from multiprocessing import Pool

from pySpatialTools.utils.util_external import Processer


class SpatialDescriptorModelProcess(Processer):
    """Spatial descriptor model processer."""
    proc_name = "Spatial descriptor model computation"

    def __init__(self, spdescmodel, logfile, lim_rows=0, n_procs=0,
                 prompt_inform=False):
        """The spatialdescriptor model process instantiation.

        Parameters
        ----------
        logfile: str
            the file we want to log all the process.
        lim_rows: int (default=0)
            the limit number of rows uninformed. If is 0, there are not
            partial information of the process.
        n_procs: int (default=0)
            the number of cpu used.
        prompt_inform: boolean (default=False)
            if we want to show the logging information in the terminal.

        """
        self._initialization()
        # Logfile
        self.logfile = logfile
        # Main class
        self.sp_descriptormodel = spdescmodel
        # Other parameters
        self.lim_rows = lim_rows
        self.bool_inform = True if self.lim_rows != 0 else False
        self.prompt_inform = prompt_inform
        self.n_procs = n_procs
        self.proc_desc = "Computation %s with %s"

    def compute_measure(self):
        """Main computation. This function acts as swicher between the
        different possibilities:
        - Parallel
        - Sequential

        Returns
        -------
        measure: np.ndarray or list
            the measure computed by the whole spatial descriptor model.

        """
        ## 0. Setting needed variables (TODO: cambiar sptypemodel)
        m_aux0 = self.sp_descriptormodel.featurers._maps_vals_i.sptype
        name_desc = self.sp_descriptormodel.name_desc
        self.proc_desc = self.proc_desc % (m_aux0, name_desc)
        t00 = self.setting_global_process()

        if self.n_procs in [-1, 0, 1, None]:
            measure = self._compute_sequential()
        else:
            measure = self._compute_parallel()

        ## 1. Closing process
        # Formatting result
        measure =\
            self.sp_descriptormodel.featurers.to_complete_measure(measure)
        # Stop tracking
        self.close_process(t00)
        return measure

    def _compute_sequential(self):
        """Main function for building the index of the selected model.

        Returns
        -------
        measure: np.ndarray or list
            the measure computed by the whole spatial descriptor model.

        """
        ## 1. Computation of the measure (parallel if)
        measure = self.sp_descriptormodel.featurers.initialization_output()
        i = 0
        # Begin to track the process
        t0, bun = self.setting_loop(self.sp_descriptormodel.n_inputs)
        for desc_i, vals_i in self.sp_descriptormodel.compute_nets_i():
            measure = self.sp_descriptormodel.featurers.\
                add2result(measure, desc_i, vals_i)
            ## Finish to track this process
            t0, bun = self.messaging_loop(i, t0, bun)
            i += 1
        return measure

    def _compute_parallel(self):
        """Compute in parallel.

        Returns
        -------
        measure: np.ndarray or list
            the measure computed by the whole spatial descriptor model.

        """
        ## Compute slicers
        idxs = create_indices_slicers(self.sp_descriptormodel._pos_inputs,
                                      self.n_procs)
        ## Copies
        spdescs = []
        for i in range(self.n_procs):
            aux_sp = copy(self)
            aux_sp.n_procs = 1
            aux_sp.set_loop(idxs[i])
            spdescs.append(aux_sp)

        measure = computer_function(spdescs, self.n_procs)
        # Joinning result
        return measure


def create_indices_slicers(main_slicer, n_procs):
    """Create slicers for individual split units."""
    start, stop, step = main_slicer.start, main_slicer.stop, main_slicer.step
    limits = [start+stop/n_procs*(i+1) for i in range(n_procs-1)]
    limits = [start] + limits + [stop]
    limits = [[limits[i], limits[i+1]] for i in range(n_procs)]
    idxs = [slice(l[0], l[1], step) for l in limits]
    return idxs


def computer_function(spdescs, n_procs):
    """Compute measure in a parallel fashion using classes."""
    p = Pool(n_procs)
    descs = p.apply_async(computer_function_i, spdescs)
    return descs


def computer_function_i(spdesc):
    """Compute sequential function."""
    return spdesc._compute_sequential()
