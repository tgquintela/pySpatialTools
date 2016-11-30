
"""
filter perturbations
--------------------
Auxiliary functions to filter perturbations.

"""

from perturbations import NonePerturbation


def sp_general_filter_perturbations(perturbations):
    """General function to filter perturbations and get the specific
    transformed perturbations for retrievermanager and featuremanager.

    Parameters
    ----------
    perturbations: pst.BasePerturbation or list
        the perturbations applied to that features.

    Returns
    -------
    perturb_ret: pst.BasePerturbation or list
        the perturbation of retriever.
    perturb_feat: pst.BasePerturbation or list
        the perturbation of features.

    """
    ## 0. Format properly the perturbations
    if type(perturbations) != list:
        perturbations = [perturbations]

    ## Perturbations
    perturb_ret = ret_filter_perturbations(perturbations)
    perturb_feat = feat_filter_perturbations(perturbations)

    return perturb_ret, perturb_feat


def ret_filter_perturbations(perturbations):
    """Filter perturbations for retriever tasks.

    Parameters
    ----------
    perturbations: pst.BasePerturbation or list
        the perturbations applied to that features.

    Returns
    -------
    perturbations: pst.BasePerturbation or list
        the perturbations applied to that features.

    """
    ## 0. Format properly the perturbations
    if type(perturbations) != list:
        perturbations = [perturbations]

    ## 1. Adapted perturbations
    dim_k_perturbs = [p.k_perturb for p in perturbations]
    type_perturbs = [p._categorytype == 'feature' for p in perturbations]
    if all(type_perturbs):
        perturbations = [NonePerturbation(sum(dim_k_perturbs))]
    else:
        for i in range(len(perturbations)):
            if perturbations[i]._categorytype == 'feature':
                perturbations[i] = NonePerturbation(perturbations[i].k_perturb)
    return perturbations


def feat_filter_perturbations(perturbations):
    """Filter perturbations for features tasks.

    Parameters
    ----------
    perturbations: pst.BasePerturbation or list
        the perturbations applied to that features.

    Returns
    -------
    perturbations: pst.BasePerturbation or list
        the perturbations applied to that features.

    """
    ## 0. Format properly the perturbations
    if type(perturbations) != list:
        perturbations = [perturbations]

    ## 1. Adapted perturbations
    dim_k_perturbs = [p.k_perturb for p in perturbations]
    type_perturbs = [p._categorytype == 'location' for p in perturbations]
    if all(type_perturbs):
        perturbations = [NonePerturbation(sum(dim_k_perturbs))]
    else:
        for i in range(len(perturbations)):
            if perturbations[i]._categorytype == 'location':
                perturbations[i] = NonePerturbation(perturbations[i].k_perturb)
    return perturbations
