
"""
Check descriptors
-----------------
Module which task is group the fucntions for testing and measuring how good
are the descriptors.

Examples
--------
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf = clf.fit(X, y)
p = clf.predict(X)
conf_mat = confusion_matrix_pred(p, y)
score = conf_mat.diagonal().sum()/conf_mat.sum()


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X, y)
probs = clf.predict_proba(X)
conf_mat = confusion_matrix_pred(probs, y)
score = conf_mat.diagonal().sum()/conf_mat.sum()

"""

from sklearn.cross_validation import cross_val_score


def test_descriptors(descriptors, feat_arr, skmodel):
    "Test how good are the descriptors for "
    predicted_probs = skmodel.fit_predict_proba(descriptors, feat_arr.ravel())
    conf_mat = confusion_matrix_creation(predicted_probs, feat_arr.ravel())
    score = conf_mat.diagonal().sum()/conf_mat.sum()
    return score


def confusion_matrix_probs(predicted_probs, feat_arr):
    "Confusion matrix from a matrix of probabilities."
    vals = np.unique(feat_arr)
    n_vals = vals.shape[0]
    feat_arr = feat_arr.ravel()

    conf_mat = np.zeros((n_vals, n_vals))
    for i in xrange(predicted_probs.shape[0]):
        conf_mat[feat_arr[i], :] += predicted_probs[i, :]
    # Normalization
    for i in range(n_vals):
        conf_mat[i, :] = conf_mat[i, :]/(feat_arr == vals[i]).sum()
    return conf_mat

def confusion_matrix_pred(predicted_labels, feat_arr):
    "Confusion matrix from predictions of the labels."
    vals = np.unique(feat_arr)
    n_vals = vals.shape[0]
    feat_arr = feat_arr.ravel()

    conf_mat = np.zeros((n_vals, n_vals))
    for i in xrange(predicted_probs.shape[0]):
        conf_mat[feat_arr[i], :] += predicted_probs[i, :]
    # Normalization
    for i in range(n_vals):
        conf_mat[i, :] = conf_mat[i, :]/(feat_arr == vals[i]).sum()
    return conf_mat
