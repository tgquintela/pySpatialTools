
"""
artificial measure
------------------
Creation of artificial measure
"""

import numpy as np


############################### Create measure ################################
###############################################################################
def create_artificial_measure_array(n_k, n_vals_i, n_feats):
    measure = np.random.random((n_vals_i, n_feats, n_k))
    return measure


def create_artificial_measure_append(n_k, n_vals_i, n_feats):
    rounds = np.random.randint(0, 40)
    measure = create_empty_append(n_k, n_vals_i, n_feats)
    for i in range(rounds):
        n_iss = np.random.randint(0, 10)
        vals_i = create_vals_i(n_iss, n_vals_i, n_k)
        x_i = create_features_i_dict(n_feats, n_iss, n_k)
        for k in range(len(vals_i)):
            for i in range(len(vals_i[k])):
                measure[k][vals_i[k][i]].append(x_i[k][i])
    return measure


def create_artificial_measure_replacelist(n_k, n_vals_i, n_feats,
                                          unique_=False):
    last = 0
    rounds = np.random.randint(0, 40)
    measure = create_empty_replacelist(n_k, n_vals_i, n_feats)
    for i in range(rounds):
        n_iss = np.random.randint(0, 10)
        if unique_:
            vals_i = np.array([last+np.arange(n_iss)]*n_k)
            last += n_iss
        else:
            vals_i = create_vals_i(n_iss, n_vals_i, n_k)
        x_i = create_features_i_dict(n_feats, n_iss, n_k)
        for k in range(len(vals_i)):
            measure[k][0].append(x_i[k])
            measure[k][1].append(vals_i[k])
    return measure


############################### Empty measure #################################
###############################################################################
def create_empty_array(n_k, n_vals_i, n_feats):
    return np.zeros((n_vals_i, n_feats, n_k))


def create_empty_append(n_k, n_iss, n_feats):
    return [[[]]*n_iss]*n_k


def create_empty_replacelist(n_k, n_iss, n_feats):
    return [[[], []]]*n_k


############################### Vals_i creation ###############################
###############################################################################
def create_vals_i(n_iss, nvals, n_k):
    return np.random.randint(0, nvals, n_iss*n_k).reshape((n_k, n_iss))


############################### Empty features ################################
###############################################################################
def create_empty_features_array(n_feats, n_iss, n_k):
    return np.zeros((n_k, n_iss, n_feats))


def create_empty_features_dict(n_feats, n_iss, n_k):
    return [[{}]*n_iss]*n_k


################################ X_i features #################################
###############################################################################
def create_features_i_array(n_feats, n_iss, n_k):
    x_i = np.random.random((n_k, n_iss, n_feats))
    return x_i


def create_features_i_dict(n_feats, n_iss, n_k):
    x_i = []
    for k in range(n_k):
        x_i_k = []
        for i in range(n_iss):
            keys = np.unique(np.random.randint(1, n_feats, n_feats))
            keys = [str(e) for e in keys]
            values = np.random.random(len(keys))
            x_i_k.append(dict(zip(keys, values)))
        x_i.append(x_i_k)
    return x_i
