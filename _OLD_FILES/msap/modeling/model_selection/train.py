# -*- coding: utf-8 -*-
"""Model training module.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd


def train_grid_search_cv(clf, X, y, param_grid, splits=None):
    """Performs grid search for the given classifier.

    Args:
        clf (imblearn.Pipeline): A classifier object with `fit` method.
        X (pd.DataFrame): Input data.
        y (pd.Series): Target data.
        param_grid (dict): Parameters for grid search.
        splits (list): A list of splits from StratifiedKFold.

    Returns:
        result (dict): The grid search CV results and best model stats.

    """
    def tn_score(y_true, y_pred):
        return confusion_matrix(y_true, y_pred).ravel()[0]

    def fp_score(y_true, y_pred):
        return confusion_matrix(y_true, y_pred).ravel()[1]

    def fn_score(y_true, y_pred):
        return confusion_matrix(y_true, y_pred).ravel()[2]

    def tp_score(y_true, y_pred):
        return confusion_matrix(y_true, y_pred).ravel()[3]

    result = {}

    grid_search_cv = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        refit='f1',
        scoring={
            'tn': make_scorer(tn_score),
            'fp': make_scorer(fp_score),
            'fn': make_scorer(fn_score),
            'tp': make_scorer(tp_score),
            'f1': make_scorer(f1_score)
        },
        n_jobs=-1,
        cv=splits)
    grid_search_cv.fit(X, y)

    # Recording confusion matrix for each fold each param set.
    cv_results = grid_search_cv.cv_results_
    for i, param in enumerate(cv_results['params']):
        result[i] = {
            'param': param,
        }
        for i_split in range(len(splits)):
            result[i][f'split_{i_split}'] = {}
            for score in ['tn', 'fp', 'fn', 'tp', 'f1']:
                result[i][f'split_{i_split}'][score] \
                    = cv_results[f'split{i_split}_test_{score}'][i]

    # Recording the best hyperparameters and stats.
    result['best'] = result[grid_search_cv.best_index_]

    return result


def train_cv(clf, X, y, splits=None):
    """Performs only CV for the given classifier.

    Args:
        clf (imblearn.Pipeline): A classifier object with `fit` method.
        X (pd.DataFrame): Input data.
        y (pd.Series): Target data.
        splits (list): A list of splits from StratifiedKFold.


    Returns:
        result (dict): The grid search CV results and best model stats.

    """
    if splits is None:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True).split(X, y))

    result = {}
    y_trues = []
    y_preds = []
    y_probs = []
    tns = []
    fps = []
    fns = []
    tps = []
    f1s = []

    for idx_train, idx_test in splits:
        X_train = X.iloc[idx_train, :]
        X_test = X.iloc[idx_test, :]
        y_train = y.iloc[idx_train]
        y_test = y.iloc[idx_test]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = pd.DataFrame(
            clf.predict_proba(X_test),
            columns=clf.classes_)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        y_trues += [y_test]
        y_preds += [y_pred]
        y_probs += [y_prob]
        tns += [tn]
        fps += [fp]
        fns += [fn]
        tps += [tp]
        f1s += [f1_score(y_test, y_pred)]

    # Record the only one hyperparameter set.
    result[0] = {
        'param': None,
    }
    for i_split in range(len(splits)):
        result[0][f'split_{i_split}'] = {
            'tn': tns[i_split],
            'fp': fps[i_split],
            'fn': fns[i_split],
            'tp': tps[i_split],
            'f1': f1s[i_split],
        }

    # Record the best hyperparameter set, which is the only one
    #   hyperparameter set.
    result['best'] = result[0]

    return result
