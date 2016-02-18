
"""
Module with functions to evaluate temporal evolution of different variables
and its distributions.
"""

import pandas as pd
import numpy as np

from ..Plotting import general_plot


def compute_tmpdist_describe(df, info_var):
    """"""
    summary = info_var
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)
    return summary
