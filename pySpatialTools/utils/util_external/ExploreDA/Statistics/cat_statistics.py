
"""
Module which groups all the functions needed to compute the statistics and the
description of the categorical variables.

"""

from ..Plotting import general_plot


def compute_cat_describe(df, info_var):
    """Function created to aggregate all the exploratory information of
    variable studied.
    """

    ## 0. Needed variables
    if info_var['variables'] == list:
        variable = info_var['variables'][0]
    else:
        variable = info_var['variables']

    ## 1. Summary
    summary = info_var
    summary['count_table'] = cat_count(df, variable)
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)

    return summary


## Categorical count
def cat_count(df, variable):
    if type(variable) == list:
        variable = variable[0]
    counts = df[variable].value_counts()
    return counts
