
"""
Module which groups all the related function of the compute of statistics in
coordinates data.
"""

from ..Plotting import general_plot


def compute_coord_describe(df, info_var):
    """"""
    summary = info_var
    # Plot
    if info_var['ifplot'] in [True, 'True', 'TRUE']:
        summary['plots'] = general_plot(df, info_var)
    return summary


def mean_coord_by_values(df, coordinates_vars, var2agg):
    """Compute the average positions for the values of a variable."""
    table = df.pivot_table(rows=var2agg, values=coordinates_vars)
    return table
