
"""
utils
-----
Module which groups the functions related with spatial utils which can be
useful.

"""

import numpy as np


def check_discretizors(discretizor):
    """Function which acts as a checker of discretization object.

    Compulsary requisits
    --------------------
    For each instantiable son classes of spatial discretizers, are required the
    next functions and parameters:
    - _map_loc2regionid (function)
    - map_regionid2regionlocs (function)
    - _compute_limits (function)
    - _compute_contiguity_geom (function)
    - get_nregs (function)
    - get_regions_id (function)
    - get_regionslocs (function)

    - limits (parameter)
    - borders (parameter)
    - regionlocs (parameter)
    - regions_id (parameter)
    - n_dim (parameter) [0, 1, 2, 3, 'n']
    - metric (parameter) [True, False]
    - format_ (parameter) ['explicit', 'implicit']
    - multiple (parameter) [True, False]

    """
    ## 0. Functions needed
    def check_requireds(requisits, actual):
        fails = []
        for req in requisits:
            if req not in actual:
                fails.append(req)
        logi = bool(fails)
        return logi, fails

    def creation_mesage(fails, types):
        msg = """The following %s which are required are not in the discretizor
        given by the user: %s."""
        msg = msg % (types, str(fails))
        return msg

    ## 1. Constraints
    lista = dir(discretizor)
    required_p = ['n_dim', 'metric', 'format_']
    required_f = ['_compute_limits', '_compute_contiguity_geom',
                  '_map_loc2regionid', '_map_regionid2regionlocs']

    ## 2. Checking constraints
    logi_p, fails_p = check_requireds(required_p, lista)
    logi_f, fails_f = check_requireds(required_f, lista)

    ## 3. Raise Error if it is needed
    if logi_p or logi_f:
        if logi_p and logi_f:
            msg = creation_mesage(fails_p, 'parameters')
            msg = msg + "\n"
            msg += creation_mesage(fails_f, 'functions')
        elif logi_p:
            msg = creation_mesage(fails_p, 'parameters')
        elif logi_f:
            msg = creation_mesage(fails_f, 'functions')
        raise TypeError(msg)


def check_flag_multi(regions):
    """Check if there is multiple region assignation in regions."""
    if type(regions) == np.ndarray:
        flag_multi = False
    else:
        try:
            regions = np.array(regions)
            flag_multi = False
        except:
            flag_multi = True
    return flag_multi

#from pySpatialTools.Retrieve.Spatial_Relations import format_out_relations
#        if region_id is None:
#            contiguity = format_out_relations(contiguity, out_)
#        out_: optional ['sparse', 'list', 'network', 'sp_relations']
#            how to present the results.

#from scipy.spatial.distance import cdist
#from sklearn.neighbors import KDTree
#from pythonUtils.parallel_tools import distribute_tasks
