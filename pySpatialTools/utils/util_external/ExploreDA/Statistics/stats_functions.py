
"""
Calcular estadistiques.
"""


import numpy as np
import pandas as pd

from cat_statistics import compute_cat_describe
from cont_statistics import compute_cont_describe
from coord_statistics import compute_coord_describe
from temp_statistics import compute_temp_describe
from tmpdist_statistics import compute_tmpdist_describe


def compute_stats(df, info_var):
    """"""

    typevar = info_var['type'].lower()
    if typevar in ['discrete', 'categorical']:
        stats = compute_cat_describe(df, info_var)
    elif typevar == 'continuous':
        stats = compute_cont_describe(df, info_var)
    elif typevar == 'coordinates':
        stats = compute_coord_describe(df, info_var)
    elif typevar in ['time', 'temporal']:
        stats = compute_temp_describe(df, info_var)
    elif typevar == 'tmpdist':
        stats = compute_tmpdist_describe(df, info_var)
    else:
        print typevar, info_var['variables']
    return stats


## Creation of cnae index at a given level
def cnae_index_level(col_cnae, level):
    pass


# Distance
def distance_cnae(col_cnae):
    pass


def finantial_per_year(servicios):
    """Function which transform the servicios data to a data for each year.
    """
    pass


## Translate to latex summary
## Individually by variable
def extend_data_by_year(df, variables, newvars=None):
    """
    """
    #year_formation = lambda x: '0'*(2-len(str(x)))+str(x)
    #years = [year_formation(i) for i in range(6, 13)] if years is None else
    #years

    if newvars is None:
        newvars = [variables[i][0][2:] for i in range(len(variables))]

    df2 = []
    y = len(variables[0])
    for i in range(y):
        aux = df[[variables[j][i] for j in range(len(variables))]]
        aux.columns = newvars
        df2.append(aux)
    df2 = pd.concat(df2)

    return df2


###############################################################################
###############################################################################
