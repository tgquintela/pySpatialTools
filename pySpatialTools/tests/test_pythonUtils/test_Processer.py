
"""
Test processer
--------------
Test for the processer module.

"""

import os
#from parallel_tools import distribute_tasks, reshape_limits
from pySpatialTools.utils.util_external import Logger, Processer


class TesterProcesserClass(Processer):
    proc_name = "Tester Processer"

    def __init__(self, logfile, lim_rows=0, n_procs=0,
                 prompt_inform=False):
        self._initialization()
        # Logfile
        self.logfile = logfile
        # Other parameters
        self.lim_rows = lim_rows
        self.bool_inform = True if self.lim_rows != 0 else False
        self.prompt_inform = prompt_inform
        self.n_procs = n_procs
        self.proc_desc = "Computation %s with %s"
        self.create_subprocess_hierharchy([['prueba']])

    def compute(self):
        # Main function to test the main utilities
        t00 = self.setting_global_process()
        # Begin to track the process
        t0, bun = self.setting_loop(100)
        t0_s = self.set_subprocess([0, 0])
        for i in range(100):
            ## Finish to track this process
            t0, bun = self.messaging_loop(i, t0, bun)
            i += 1
        self.close_subprocess([0, 0], t0_s)
        self.save_process_info('prueba')
        self.close_process(t00)


def test():
    ## Prepare the process
    logfile = Logger('logfile.log')
    proc = TesterProcesserClass(logfile)
    ## Process
    proc.compute()
    ## Remove the files created
    os.remove('logfile.log')
    os.remove('prueba')
