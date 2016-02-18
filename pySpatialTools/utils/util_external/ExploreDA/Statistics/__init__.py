
"""
Module oriented to compute the desired statistical description of each
selected variables or bunch of variables with the selected method.
"""

import pandas as pd
import numpy as np
from stats_functions import compute_stats
from utils import clean_dict_stats
import time

from ..IO.parse_data import parse_instructions_file
from ..IO.output_to_latex import describe2latex

from pythonUtils.Logger import Logger


########### Global variables needed
##################################################################
message0 = """========================================
Start computing stats:
----------------------
(%s)

"""
message1 = "Computing stats ...\n"
message1a = "Stats of variable %s: "
message1b = "Exporting to tex: "
message2 = "completed in %f seconds.\n"
message3 = "Total time expended stats process: %f seconds.\n"
message_close = '----------------------------------------\n'


########### Class for parsing
##################################################################
class Statistics():
    """The object which performs the computation of the statistics.

    TODO
    ----
    Check if the variables in the info are in the dataframe and act in
    consequence.
    create plots function to create from stats the plots.
    - Transform this class as a Processer

    """

    def __init__(self, fileinstructions, study_info={}, logfile=None):
        '''Initialization of the stats computation.'''
        describ_info = parse_instructions_file(fileinstructions)
        self.fileinstructions = fileinstructions
        self.info = describ_info
        self.stats = None
        self.study_info = study_info
        self.logfile = Logger(logfile)

    def compute_stats(self, dataframe, info=None):
        '''Function to compute the statistics for all the columns.'''
        ## 0. Prepare inputs
        self.info = self.info if info is None else info
        # Tracking process with logfile
        t00 = time.time()
        self.logfile.write_log(message0 % self.fileinstructions)
        self.logfile.write_log(message1)
        ## 1. Compute stats
        stats = []
        for i in self.info.index:
            info_var = dict(self.info.iloc[i])
            # Tracking process with logfile
            t0 = time.time()
            self.logfile.write_log(message1a % info_var['variables'])
            # Computing stats of the i-th variable
            stats.append(compute_stats(dataframe, info_var))
            # Stop tracking the process
            self.logfile.write_log(message2 % (time.time()-t0))
        ## 2. Save and return
        self.stats = stats
        # TODO: Order by column order!!!
        countsnull = np.sum(dataframe.notnull())
        aux = pd.DataFrame([countsnull, dataframe.shape[0]-countsnull],
                           columns=['non-null', 'null'])
        self.study_info['global_stats'] = aux
        # Stop tracking the process
        self.logfile.write_log(message3 % (time.time()-t00))
        self.logfile.write_log(message_close)
        return stats

    def to_latex(self, filepath=None):
        ## Tracking the process
        t0 = time.time()
        self.logfile.write_log(message1b)
        ## 1. Compute transformation
        doc = describe2latex(self.study_info, self.stats)
        ## 2. Write output
        if filepath is None:
            return doc
        else:
            #Write doc
            with open(filepath, 'w') as myfile:
                myfile.write(doc)
            myfile.close()
        # Stop tracking the process
        self.logfile.write_log(message2 % (time.time()-t0))
        self.logfile.write_log(message_close)

    def clean_stats(self, stats=None):
        if stats is None:
            stats = self.stats
        stats = clean_dict_stats(stats)
        return stats
