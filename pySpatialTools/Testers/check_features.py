
"""
Check descriptors
-----------------
Module which task is group the functions for testing and measuring how good
are the descriptor for representing our data.

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix


def fit_model(model, X, y):
    "Function to fit the model we want."
    n_folds = 3
    skf = StratifiedKFold(y, n_folds=n_folds)
    models, measures = [], []
    for train_index, test_index in skf:
        ## Extract Kfold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ## Fit models
        model_i = model()
        models.append(model_i.fit(X_train, y_train))
        ## Compute measure
        proba_m = model_i.predict_proba(X_test)
        measures.append(compute_measure(y_test, proba_m))

    i = np.argmax(measures)
    model, measure = models[i], measures[i]
    return model, measure


def descriptors_quality(model, X, y):
    "Compute the quality measure of the descriptors."
    proba_m = model.predict_proba(X)
    measure = compute_measure(y, proba_m)
    return measure


def compute_measure(real, pred):
    "Compute measure of performance from the predictions."
    conf_mat = compute_confusion_matrix(real, pred)
    score = from_confmat2score(conf_mat)
    return score


def from_confmat2score(conf_mat, method='accuracy', comp=None):
    "Compute the score from confusion matrix."
    if type(method) == str:
        if method == 'accuracy':
            score = conf_mat.diagonal().sum()/conf_mat.sum()
    elif type(method).__name__ == 'function':
        score = method(conf_mat)
    return score


def compute_confusion_matrix(real, pred, normalization=None):
    "Compute confusion matrix."
    if real.shape == pred.shape:
        conf_mat = confusion_matrix(real, pred)
    else:
        conf_mat = confusion_matrix_probs(pred, real)
    return conf_mat


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
