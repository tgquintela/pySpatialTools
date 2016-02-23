
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
        """
        ## 0. Setting needed variables (TODO: cambiar sptypemodel)
#        m_aux0 = self.sp_descriptormodel.sptypemodel
        m_aux0 = "Matrix"  # Temporal
        name_desc = self.sp_descriptormodel.name_desc
        self.proc_desc = self.proc_desc % (m_aux0, name_desc)
        t00 = self.setting_global_process()

        if self.n_procs in [-1, 0, 1, None]:
            desc = self._compute_sequential()
        else:
            desc = self._compute_parallel()

        ## 1. Closing process
        # Formatting result
        desc = self.sp_descriptormodel.featurers.to_complete_measure(desc)
        # Stop tracking
        self.close_process(t00)
        return desc

    def _compute_sequential(self):
        """Main function for building the index of the selected model."""
        ## 1. Computation of the measure (parallel if)
        desc = self.sp_descriptormodel.featurers.initialization_output()
        i = 0
        # Begin to track the process
        t0, bun = self.setting_loop(self.sp_descriptormodel.n_inputs)
        for desc_i, vals_i in self.sp_descriptormodel.compute_nets_i():
            for k in range(len(vals_i)):
                desc[vals_i[k], :, k] = self.sp_descriptormodel.featurers.\
                    add2result(desc[vals_i[k], :, k], desc_i[k])
            ## Finish to track this process
            t0, bun = self.messaging_loop(i, t0, bun)
            i += 1
        return desc

    def _compute_parallel(self):
        """Compute in parallel."""
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

        desc = computer_function(spdescs, self.n_procs)
        # Joinning result
        return desc


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
