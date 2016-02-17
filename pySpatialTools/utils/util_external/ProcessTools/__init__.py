

"""
Module which contains the abstract class of a process.
It generalize a common process in order to be easier to compute with tools
as display information of the time of the process and other things.

TODO:
-----
- Generalize the messaging
- Compulsary inputs
- Process with pipelines and subprocesses
"""

import shelve
import time


###############################################################################
########### Global variables needed for this module
###############################################################################
message0 = """========================================
Start process %s:
--------------------
(%s)

"""
message1 = "%s: "
message2 = "completed in %f seconds.\n"
message_init_loop = "Total number of iterations to compute: %s"
message_loop = " %s bunch of %s iterations completed in %f seconds.\n"

message_close0 = '-'*70
message_last = "Total time expended computing the process: %f seconds.\n"
message_close = '-'*70+'\n'
###############################################################################
###############################################################################


###############################################################################
################################## MAIN CLASS #################################
###############################################################################
class Processer():
    """Abstract class for the processers some computations.
    """

    ### Class parameters
    ## Process descriptors
    time_expended = 0.  # Time expended along the process
    t_expended_subproc = []  # Time expended in each subprocesses
    n_procs = 0  # Number of cpu used in parallelization (0 no parallel)
    proc_name = ""  # Name of the process
    proc_desc = ""  # Process description
    subproc_desc = []  # Subprocess description
    ## Logger info
    lim_rows = 0  # Lim of rows done in a bunch. For matrix comp or information
    logfile = None  # Log file
    ## Bool options
    bool_inform = False  # Give information of the process

    def save_process_info(self, outputfile):
        database = shelve.open(outputfile)
        out = self.to_dict()
        for e in out.keys():
            database[e] = out[e]
        database.close()

    def to_dict(self):
        "Transform the class information into a dictionary."
        out = {'time_expended': self.time_expended, 'n_procs': self.n_procs,
               'proc_name': self.proc_name, 'lim_rows': self.lim_rows,
               'logfile': self.logfile}
        return out

    def setting_loop(self, N_t):
        self.logfile.write_log(message_init_loop % str(N_t))
        t0, bun = time.time(), 0
        return t0, bun

    def messaging_loop(self, i, t0, bun):
        "Message into the loop."
        if self.bool_inform and (i % self.lim_rows) == 0 and i != 0:
            t_sp = time.time()-t0
            bun += 1
            self.logfile.write_log(message_loop % (bun, self.lim_rows, t_sp))
            t0 = time.time()
        return t0, bun

    def close_process(self, t00):
        "Closing process."
        ## Closing process
        t_expended = time.time()-t00
        self.logfile.write_log(message_close0)
        self.logfile.write_log(message_last % t_expended)
        self.logfile.write_log(message_close)
        self.time_expended = t_expended

    def setting_global_process(self):
        "Setting up the process."
        ## Initiating process
        message0 = initial_message_creation(self.proc_name, self.proc_desc)
        self.logfile.write_log(message0)
        t00 = time.time()
        return t00

    def set_subprocess(self, index_sub):
        aux = self.subproc_desc
        for i in index_sub:
            aux = aux[i]
        message1 % aux
        t0 = time.time()
        return t0

    def close_subprocess(self, index_sub, t0):
        ## Save time
        t_expended = time.time()-t0
        if len(index_sub) == 1:
            i0 = index_sub[0]
            self.t_expended_subproc[i0] = t_expended
            proc_name = self.subproc_desc[i0]
        elif len(index_sub) == 2:
            i0, i1 = index_sub[0], index_sub[1]
            self.t_expended_subproc[i0][i1] = t_expended
            proc_name = self.subproc_desc[i0][i1]
        elif len(index_sub) == 3:
            i0, i1, i2 = index_sub[0], index_sub[1], index_sub[2]
            self.t_expended_subproc[i0][i1][i2] = t_expended
            proc_name = self.subproc_desc[i0][i1][i2]
        elif len(index_sub) == 4:
            i0, i1, i2 = index_sub[0], index_sub[1], index_sub[2]
            i3 = index_sub[3]
            self.t_expended_subproc[i0][i1][i2][i3] = t_expended
            proc_name = self.subproc_desc[i0][i1][i2][i3]
        elif len(index_sub) == 5:            
            i0, i1, i2 = index_sub[0], index_sub[1], index_sub[2]
            i3, i4 = index_sub[3], index_sub[4]
            self.t_expended_subproc[i0][i1][i2][i3][i4] = t_expended
            proc_name = self.subproc_desc[i0][i1][i2][i3][i4]
        elif len(index_sub) == 6:            
            i0, i1, i2 = index_sub[0], index_sub[1], index_sub[2]
            i0, i1, i2 = index_sub[3], index_sub[4], index_sub[5]
            self.t_expended_subproc[i0][i1][i2][i3][i4][i5] = t_expended
            proc_name = self.subproc_desc[i0][i1][i2][i3][i4][i5]
        ## Logfile writing
        self.logfile.write_log(message1 % proc_name)
        self.logfile.write_log(message2 % t_expended)

    def check_subprocess(self):
        "Check if the subprocess is properly formatted."
        n1 = len(self.t_expended_subproc)
        n2 = len(self.subproc_desc)
        bool_res = True
        bool_res = bool_res and (n1 == n2)
        for i in range(n2):
            if type(self.subproc_desc[i]) == list:
                aux = type(self.t_expended_subproc[i]) == list
                bool_res = bool_res and aux
                n1a = len(self.t_expended_subproc[i])
                n2a = len(self.subproc_desc[i])
                bool_res = bool_res and (n1a == n2a)
        return bool_res


def initial_message_creation(proc_name, proc_desc):
    line0 = "="*30
    line1 = "Start process %s:" % proc_name
    line2 = "-" * len(line1)
    line3 = "(%s)" % proc_desc
    return line0+"\n"+line1+"\n"+line2+"\n"+line3+"\n\n"
