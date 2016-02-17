
"""
Metric utilities
----------------
Utilities to define a metric .

"""


## Point elements
###############################################################################
def measure_difference(element_i, element_j, pars={}):
    measure = (element_j - element_i)
    return measure


def unidimensional_periodic(element_i, element_j, pars={}):
    """Unidimensional periodic metric."""
    periodic = pars['periodic'] if 'periodic' in pars else None
    if periodic is None:
        return measure_difference(element_i, element_j)
    ## In the same periodic dimension
    element_i = element_i % periodic
    element_j = element_j % periodic
    ## Ordering: element_j is the bigger
    if element_j < element_i:
        element_i, element_j = element_j, element_i
    ## Computing possible results
    a0 = abs(measure_difference(element_i, element_j))
    a1 = measure_difference(element_j, periodic) + element_i
    ## Get the min
    measure = a0 if a0 < a1 else a1
    return measure
