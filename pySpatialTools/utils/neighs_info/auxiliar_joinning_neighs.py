
"""
Auxiliar joinning functions
---------------------------
Auxiliar module of Neighs_info for joinning functions.
"""

import numpy as np


###############################################################################
########################### Auxiliar join functions ###########################
###############################################################################
######################### Auxiliar general functions ##########################
def check_compatibility_neighs(neighs_info0, neighs_info1):
    """Check if the different neighs_info are compatible.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.

    """
    assert(neighs_info0.ks == neighs_info1.ks)
    assert(len(neighs_info0.idxs) == len(neighs_info1.idxs))
    assert(neighs_info0.staticneighs == neighs_info1.staticneighs)
    if not neighs_info0.staticneighs:
        assert(len(neighs_info0.idxs[0]) == len(neighs_info1.idxs[0]))
    none_sprelpos0 = neighs_info0.sp_relative_pos is None
    none_sprelpos1 = neighs_info1.sp_relative_pos is None
    assert(none_sprelpos0 == none_sprelpos1)
    assert(neighs_info0.ifdistance == neighs_info1.ifdistance)
    if neighs_info0.ifdistance:
        assert(not none_sprelpos0)


def get_ki_info_static_dist(neighs_info0, neighs_info1):
    """Iteration which generates from the neighs_info from a staticneighs with
    defined sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.

    Returns
    -------
    ks_k: int
        the perturbation indices.
    iss_i: int
        the element indices.
    neighs0_i: list or np.ndarray
        the neighs for each perturbation `k` and element `i` of neighbourhood0.
    neighs1_i: list or np.ndarray
        the neighs for each perturbation `k` and element `i` of neighbourhood1.
    rel_pos0_i: list or np.ndarray
        the relative position for each perturbation `k` and element `i` of
        neighbourhood0.
    rel_pos1_i: list or np.ndarray
        the relative position for each perturbation `k` and element `i` of
        neighbourhood1.

    """
    iss = neighs_info0.iss
    for i in range(len(iss)):
        yield iss[i], neighs_info0.idxs[i], neighs_info1.idxs[i],\
            neighs_info0.sp_relative_pos[i], neighs_info1.sp_relative_pos[i]


def get_ki_info_static_notdist(neighs_info0, neighs_info1):
    """Iteration which generates from the neighs_info from a staticneighs
    without defined sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.

    Returns
    -------
    iss_i: int
        the element indices.
    neighs0_i: list or np.ndarray
        the neighs for each element `i` of neighbourhood0.
    neighs1_i: list or np.ndarray
        the neighs for each element `i` of neighbourhood1.

    """
    iss = neighs_info0.iss
    for i in range(len(iss)):
        yield iss[i], neighs_info0.idxs[i], neighs_info1.idxs[i]


def get_ki_info_notstatic_dist(neighs_info0, neighs_info1):
    """Iteration which generates from the neighs_info from a not staticneighs
    with defined sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.

    Returns
    -------
    ks_k: int
        the perturbation indices.
    iss_i: int
        the element indices.
    neighs0_k_i: list or np.ndarray
        the neighs for each perturbation `k` and element `i` of neighbourhood0.
    neighs1_k_i: list or np.ndarray
        the neighs for each perturbation `k` and element `i` of neighbourhood1.
    rel_pos0_k_i: list or np.ndarray
        the relative position for each perturbation `k` and element `i` of
        neighbourhood0.
    rel_pos1_k_i: list or np.ndarray
        the relative position for each perturbation `k` and element `i` of
        neighbourhood1.

    """
    ks = neighs_info0.ks
    iss = neighs_info0.iss
    for k in range(len(ks)):
        for i in range(len(iss)):
            yield ks[k], iss[i], neighs_info0.idxs[k][i],\
                neighs_info1.idxs[k][i],\
                neighs_info0.sp_relative_pos[k][i],\
                neighs_info1.sp_relative_pos[k][i]


def get_ki_info_notstatic_notdist(neighs_info0, neighs_info1):
    """Iteration which generates from the neighs_info from a not staticneighs
    without defined sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.

    Returns
    -------
    ks_k: int
        the perturbation indices.
    iss_i: int
        the element indices.
    neighs0_k_i: list or np.ndarray
        the neighs for each perturbation `k` and element `i` of neighbourhood0.
    neighs1_k_i: list or np.ndarray
        the neighs for each perturbation `k` and element `i` of neighbourhood1.

    """
    ks = neighs_info0.ks
    iss = neighs_info0.iss
    for k in range(len(ks)):
        for i in range(len(iss)):
            yield ks[k], iss[i], neighs_info0.idxs[k][i],\
                neighs_info1.idxs[k][i]


############################ AND joining functions ############################
def join_neighsinfo_AND_general(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are in both neighbourhoods.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    joined_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 1. Possibilities
    staticneighs = neighs_info0.staticneighs
    ifdistance = neighs_info1.sp_relative_pos is not None
    if staticneighs:
        if ifdistance:
            joined_neighs_info =\
                join_neighsinfo_AND_static_dist(neighs_info0, neighs_info1,
                                                joiner_pos)
        else:
            joined_neighs_info =\
                join_neighsinfo_AND_static_notdist(neighs_info0, neighs_info1)
    else:
        if ifdistance:
            joined_neighs_info =\
                join_neighsinfo_AND_notstatic_dist(neighs_info0, neighs_info1,
                                                   joiner_pos)
        else:
            joined_neighs_info =\
                join_neighsinfo_AND_notstatic_notdist(neighs_info0,
                                                      neighs_info1)
    return joined_neighs_info


def join_neighsinfo_AND_static_dist(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are in both neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss = len(neighs_info0.iss)
    joined_idxs, joined_relpos = [[]]*n_iss, [[]]*n_iss
    sequency = get_ki_info_static_dist(neighs_info0, neighs_info1)
    for i, neighs0_ki, neighs1_ki, relpos0_ki, relpos1_ki in sequency:
        joined_idxs[i], joined_relpos[i] =\
            join_neighs_AND(neighs0_ki, neighs1_ki, relpos0_ki,
                            relpos1_ki, joiner_pos)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs, joined_relpos)
    return new_neighs_info


def join_neighsinfo_AND_notstatic_dist(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are in both neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss, n_k = len(neighs_info0.iss), len(neighs_info0.ks)
    joined_idxs, joined_relpos = [[[]]*n_iss]*n_k, [[[]]*n_iss]*n_k
    sequency = get_ki_info_notstatic_dist(neighs_info0, neighs_info1)
    for k, i, neighs0_ki, neighs1_ki, relpos0_ki, relpos1_ki in sequency:
        joined_idxs[k][i], joined_relpos[k][i] =\
            join_neighs_AND(neighs0_ki, neighs1_ki, relpos0_ki,
                            relpos1_ki, joiner_pos)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs, joined_relpos)
    return new_neighs_info


def join_neighsinfo_AND_static_notdist(neighs_info0, neighs_info1):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are in both neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss = len(neighs_info0.iss)
    joined_idxs = [[]]*n_iss
    sequency = get_ki_info_static_notdist(neighs_info0, neighs_info1)
    for i, neighs0_ki, neighs1_ki in sequency:
        joined_idxs[i] = join_neighs_AND_notrelpos(neighs0_ki, neighs1_ki)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs)
    return new_neighs_info


def join_neighsinfo_AND_notstatic_notdist(neighs_info0, neighs_info1):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are in both neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss, n_k = len(neighs_info0.iss), len(neighs_info0.ks)

    joined_idxs = [[[]]*n_iss]*n_k
    sequency = get_ki_info_notstatic_notdist(neighs_info0, neighs_info1)
    for k, i, neighs0_ki, neighs1_ki in sequency:
        joined_idxs[k][i] = join_neighs_AND_notrelpos(neighs0_ki, neighs1_ki)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs)
    return new_neighs_info


def join_neighs_AND(idxs0_ki, idxs1_ki, relpos0_ki, relpos1_ki, joiner_pos):
    """Join neighs with AND.

    Parameters
    ----------
    idxs0_ki: list or np.ndarray
        the indices of the neighs of neighbourhood0.
    idxs1_ki: list or np.ndarray
        the indices of the neighs of neighbourhood1.
    relpos0_ki: list or np.ndarray
        the relative positions of the neighs of neighbourhood0.
    relpos1_ki: list or np.ndarray
        the relative positions of the neighs of neighbourhood1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    neighs: list
        the joined neighbourhood.
    rel_pos: list
        the relative position for the joined neighbourhood.

    """
    neighs, rel_pos = [], []
    for i in range(len(idxs0_ki)):
        if idxs0_ki[i] in idxs1_ki:
            neighs.append(idxs0_ki[i])
            j = np.where(np.array(idxs1_ki) == idxs0_ki[i])[0][0]
            rel_pos.append(joiner_pos(relpos0_ki[i], relpos1_ki[j]))
    return neighs, rel_pos


def join_neighs_AND_notrelpos(idxs0_ki, idxs1_ki):
    """Join neighs with AND.

    Parameters
    ----------
    idxs0_ki: list or np.ndarray
        the indices of the neighs of neighbourhood0
    idxs1_ki: list or np.ndarray
        the indices of the neighs of neighbourhood1

    Returns
    -------
    neighs: list
        the joined neighbourhood.

    """
    neighs = []
    for i in range(len(idxs0_ki)):
        if idxs0_ki[i] in idxs1_ki:
            neighs.append(idxs0_ki[i])
    return neighs


############################# OR joining functions ############################
def join_neighsinfo_OR_general(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties keeping
    the neighs that are in one or another neighbourhood.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    joined_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 1. Possibilities
    staticneighs = neighs_info0.staticneighs
    ifdistance = neighs_info1.sp_relative_pos is not None
    if staticneighs:
        if ifdistance:
            joined_neighs_info =\
                join_neighsinfo_OR_static_dist(neighs_info0, neighs_info1,
                                               joiner_pos)
        else:
            joined_neighs_info =\
                join_neighsinfo_OR_static_notdist(neighs_info0, neighs_info1)
    else:
        if ifdistance:
            joined_neighs_info =\
                join_neighsinfo_OR_notstatic_dist(neighs_info0, neighs_info1,
                                                  joiner_pos)
        else:
            joined_neighs_info =\
                join_neighsinfo_OR_notstatic_notdist(neighs_info0,
                                                     neighs_info1)
    return joined_neighs_info


def join_neighsinfo_OR_static_dist(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties keeping
    the neighs that are in one or another neighbourhood.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss = len(neighs_info0.iss)
    joined_idxs, joined_relpos = [[]]*n_iss, [[]]*n_iss
    sequency = get_ki_info_static_dist(neighs_info0, neighs_info1)
    for i, neighs0_ki, neighs1_ki, relpos0_ki, relpos1_ki in sequency:
        joined_idxs[i], joined_relpos[i] =\
            join_neighs_OR(neighs0_ki, neighs1_ki, relpos0_ki,
                           relpos1_ki, joiner_pos)
    neighs_info0.direct_set(joined_idxs, joined_relpos)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs, joined_relpos)
    return new_neighs_info


def join_neighsinfo_OR_notstatic_dist(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties keeping
    the neighs that are in one or another neighbourhood.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss, n_k = len(neighs_info0.iss), len(neighs_info0.ks)
    joined_idxs, joined_relpos = [[[]]*n_iss]*n_k, [[[]]*n_iss]*n_k
    sequency = get_ki_info_notstatic_dist(neighs_info0, neighs_info1)
    for k, i, neighs0_ki, neighs1_ki, relpos0_ki, relpos1_ki in sequency:
        joined_idxs[k][i], joined_relpos[k][i] =\
            join_neighs_OR(neighs0_ki, neighs1_ki, relpos0_ki,
                           relpos1_ki, joiner_pos)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs, joined_relpos)
    return new_neighs_info


def join_neighsinfo_OR_static_notdist(neighs_info0, neighs_info1):
    """Join two different neighs info that shares same properties keeping
    the neighs that are in one or another neighbourhood.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss = len(neighs_info0.iss)
    joined_idxs = [[]]*n_iss
    sequency = get_ki_info_static_notdist(neighs_info0, neighs_info1)
    for i, neighs0_ki, neighs1_ki in sequency:
        joined_idxs[i] = join_neighs_OR_notrelpos(neighs0_ki, neighs1_ki)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs)
    return new_neighs_info


def join_neighsinfo_OR_notstatic_notdist(neighs_info0, neighs_info1):
    """Join two different neighs info that shares same properties keeping
    the neighs that are in one or another neighbourhood.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss, n_k = len(neighs_info0.iss), len(neighs_info0.ks)
    joined_idxs = [[[]]*n_iss]*n_k
    sequency = get_ki_info_notstatic_notdist(neighs_info0, neighs_info1)
    for k, i, neighs0_ki, neighs1_ki in sequency:
        joined_idxs[k][i] = join_neighs_OR_notrelpos(neighs0_ki, neighs1_ki)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs)
    return new_neighs_info


def join_neighs_OR(idxs0_ki, idxs1_ki, relpos0_ki, relpos1_ki, joiner_pos):
    """Join neighs with OR.

    Parameters
    ----------
    idxs0_ki: list or np.ndarray
        the indices of the neighs of neighbourhood0.
    idxs1_ki: list or np.ndarray
        the indices of the neighs of neighbourhood1.
    relpos0_ki: list or np.ndarray
        the relative positions of the neighs of neighbourhood0.
    relpos1_ki: list or np.ndarray
        the relative positions of the neighs of neighbourhood1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    neighs: list
        the joined neighbourhood.
    rel_pos: list
        the relative position for the joined neighbourhood.

    """
    neighs, rel_pos = [], []
    for i in range(len(idxs0_ki)):
        neighs.append(idxs0_ki[i])
        if idxs0_ki[i] in idxs1_ki:
            j = np.where(np.array(idxs1_ki) == idxs0_ki[i])[0][0]
            rel_pos.append(joiner_pos(relpos0_ki[i], relpos1_ki[j]))
        else:
            rel_pos.append(joiner_pos(relpos0_ki[i], None))
    for i in range(len(idxs1_ki)):
        if idxs1_ki[i] not in idxs0_ki:
            neighs.append(idxs1_ki[i])
            rel_pos.append(joiner_pos(relpos1_ki[i], None))
    return neighs, rel_pos


def join_neighs_OR_notrelpos(idxs0_ki, idxs1_ki):
    """Join neighs with OR.

    Parameters
    ----------
    idxs0_ki: list or np.ndarray
        the indices of the neighs of neighbourhood0
    idxs1_ki: list or np.ndarray
        the indices of the neighs of neighbourhood1

    Returns
    -------
    neighs: list
        the joined neighbourhood.

    """
    neighs = []
    for i in range(len(idxs0_ki)):
        neighs.append(idxs0_ki[i])
    for i in range(len(idxs1_ki)):
        if idxs1_ki[i] not in idxs0_ki:
            neighs.append(idxs1_ki[i])
    return neighs


############################ XOR joining functions ############################
def join_neighsinfo_XOR_general(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are only in one or another neighbourhoods.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    joined_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 1. Possibilities
    staticneighs = neighs_info0.staticneighs
    ifdistance = neighs_info1.sp_relative_pos is not None
    if staticneighs:
        if ifdistance:
            joined_neighs_info =\
                join_neighsinfo_XOR_static_dist(neighs_info0, neighs_info1,
                                                joiner_pos)
        else:
            joined_neighs_info =\
                join_neighsinfo_XOR_static_notdist(neighs_info0, neighs_info1)
    else:
        if ifdistance:
            joined_neighs_info =\
                join_neighsinfo_XOR_notstatic_dist(neighs_info0, neighs_info1,
                                                   joiner_pos)
        else:
            joined_neighs_info =\
                join_neighsinfo_XOR_notstatic_notdist(neighs_info0,
                                                      neighs_info1)
    return joined_neighs_info


def join_neighsinfo_XOR_static_dist(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are only in one or another neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss = len(neighs_info0.iss)
    joined_idxs, joined_relpos = [[]]*n_iss, [[]]*n_iss
    sequency = get_ki_info_static_dist(neighs_info0, neighs_info1)
    for i, neighs0_ki, neighs1_ki, relpos0_ki, relpos1_ki in sequency:
        joined_idxs[i], joined_relpos[i] =\
            join_neighs_XOR(neighs0_ki, neighs1_ki, relpos0_ki,
                            relpos1_ki, joiner_pos)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs, joined_relpos)
    return new_neighs_info


def join_neighsinfo_XOR_notstatic_dist(neighs_info0, neighs_info1, joiner_pos):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are only in one or another neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss, n_k = len(neighs_info0.iss), len(neighs_info0.ks)
    joined_idxs, joined_relpos = [[[]]*n_iss]*n_k, [[[]]*n_iss]*n_k
    sequency = get_ki_info_notstatic_dist(neighs_info0, neighs_info1)
    for k, i, neighs0_ki, neighs1_ki, relpos0_ki, relpos1_ki in sequency:
        joined_idxs[k][i], joined_relpos[k][i] =\
            join_neighs_XOR(neighs0_ki, neighs1_ki, relpos0_ki,
                            relpos1_ki, joiner_pos)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs, joined_relpos)
    return new_neighs_info


def join_neighsinfo_XOR_static_notdist(neighs_info0, neighs_info1):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are only in one or another neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss = len(neighs_info0.iss)
    joined_idxs = [[]]*n_iss
    sequency = get_ki_info_static_notdist(neighs_info0, neighs_info1)
    for i, neighs0_ki, neighs1_ki in sequency:
        joined_idxs[i] = join_neighs_XOR_notrelpos(neighs0_ki, neighs1_ki)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs)
    return new_neighs_info


def join_neighsinfo_XOR_notstatic_notdist(neighs_info0, neighs_info1):
    """Join two different neighs info that shares same properties only keeping
    the neighs that are only in one or another neighbourhoods.
    It is supposed that there are staticneighs and sp_relative_pos.

    Parameters
    ----------
    neighs_info0: pst.Neighs_Info
        the neighbourhood information of the retrieved 0.
    neighs_info1: pst.Neighs_Info
        the neighbourhood information of the retrieved 1.

    Returns
    -------
    new_neighs_info: pst.Neighs_Info
        the neighbourhood information of joined neighbourhood.

    """
    ## 0. Check the operation could be done
    check_compatibility_neighs(neighs_info0, neighs_info1)
    ## 1. Computation joinning
    n_iss, n_k = len(neighs_info0.iss), len(neighs_info0.ks)
    joined_idxs = [[[]]*n_iss]*n_k
    sequency = get_ki_info_notstatic_notdist(neighs_info0, neighs_info1)
    for k, i, neighs0_ki, neighs1_ki in sequency:
        joined_idxs[k][i] = join_neighs_XOR_notrelpos(neighs0_ki, neighs1_ki)
    new_neighs_info = neighs_info0.copy()
    new_neighs_info.direct_set(joined_idxs)
    return new_neighs_info


def join_neighs_XOR(idxs0_ki, idxs1_ki, relpos0_ki, relpos1_ki, joiner_pos):
    """Join neighs with XOR.

    Parameters
    ----------
    idxs0_ki: list or np.ndarray
        the indices of the neighs of neighbourhood0.
    idxs1_ki: list or np.ndarray
        the indices of the neighs of neighbourhood1.
    relpos0_ki: list or np.ndarray
        the relative positions of the neighs of neighbourhood0.
    relpos1_ki: list or np.ndarray
        the relative positions of the neighs of neighbourhood1.
    joiner_pos: function
        the function to join the relative positions of the different
        neighbourhood.

    Returns
    -------
    neighs: list
        the joined neighbourhood.
    rel_pos: list
        the relative position for the joined neighbourhood.

    """
    neighs, rel_pos = [], []
    for i in range(len(idxs0_ki)):
        if idxs0_ki[i] not in idxs1_ki:
            neighs.append(idxs0_ki[i])
            rel_pos.append(joiner_pos(relpos0_ki[i], None))
    for i in range(len(idxs1_ki)):
        if idxs1_ki[i] not in idxs0_ki:
            neighs.append(idxs1_ki[i])
            rel_pos.append(joiner_pos(relpos1_ki[i], None))
    return neighs, rel_pos


def join_neighs_XOR_notrelpos(idxs0_ki, idxs1_ki):
    """Join neighs with XOR.

    Parameters
    ----------
    idxs0_ki: list or np.ndarray
        the indices of the neighs of neighbourhood0
    idxs1_ki: list or np.ndarray
        the indices of the neighs of neighbourhood1

    Returns
    -------
    neighs: list
        the joined neighbourhood.

    """
    neighs = []
    for i in range(len(idxs0_ki)):
        if idxs0_ki[i] not in idxs1_ki:
            neighs.append(idxs0_ki[i])
    for i in range(len(idxs1_ki)):
        if idxs1_ki[i] not in idxs0_ki:
            neighs.append(idxs1_ki[i])
    return neighs
