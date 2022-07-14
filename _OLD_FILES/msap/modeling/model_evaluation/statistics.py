# -*- coding: utf-8 -*-
"""The module of methods that calculate statistics related to model
performance.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
from typing import Union

from scipy.spatial import distance_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix)
from sklearn.model_selection import KFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import sklearn
import numpy as np
import pandas as pd


def get_embedded_data(
        X: pd.DataFrame,
        method: str = 'tsne',
        random_state: int = None) -> np.ndarray:
    """Calculate the transformed dataset with reduced dimensionality.

    Args:
        X: The input dataset.
        method: The method for the transformation. Options = {'tsne', 'pca'}.
        random_state: The random seed.

    Returns:
        The transformed X.

    """
    N_COMPONENTS = 2

    if method == 'tsne':
        embedder = TSNE(
            n_components=N_COMPONENTS,
            learning_rate='auto',
            init='random',
            random_state=random_state)
    elif method == 'pca':
        embedder = PCA(
            n_components=N_COMPONENTS)
    else:
        raise ValueError(f"Invalid method: {method}")

    X_embedded = embedder.fit_transform(X.to_numpy())

    return X_embedded


def get_selected_features(
        clf: sklearn.base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        splits: Union[int, list[list, list]] = None) -> SFS:
    """Select features based on the Recursive Feature Elimination.

    Args:
        clf: A Scikit-learn classifier.
        X: The input dataset.
        y: The input labels.
        splits: An integer indicating the number of folds or a pre-defined
            list of training and validation split row indices.

    Returns:
        The SFS object with RFE result.

    """
    sfs = SFS(
        clf,
        k_features='parsimonious',
        forward=False,
        floating=False,
        verbose=0,
        scoring='f1',
        n_jobs=-1,
        cv=splits)
    sfs = sfs.fit(X, y)

    return sfs


def get_curve_metrics(
        clf: sklearn.base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'pr',
        splits: Union[int, list[list, list]] = None) -> dict:
    """Compute curve metrics with cross-validation.

    Args:
        clf: A Scikit-learn classifier.
        X: The input dataset.
        y: The input labels.
        method: The specification for either precision-recall curve or receiver
            operating chracteristic curve. Options = {'pr', 'roc'}.

    Returns:
        The dictionary containing curve metrics for each fold.

    """
    if method == 'pr':
        curve_fn = precision_recall_curve
        auc_fn = average_precision_score
    elif method == 'roc':
        curve_fn = roc_curve
        auc_fn = roc_auc_score
    else:
        raise ValueError(f"Invalid method {method}")

    curves = {}
    if splits is None:
        splits = KFold(n_splits=5, shuffle=True).split(X)

    # Get curve metrics for each fold.
    y_scores = []
    y_tests = []
    for i, split in enumerate(splits):
        X_train, y_train = X.loc[split[0]], y.loc[split[0]]
        X_test, y_test = X.loc[split[1]], y.loc[split[1]]

        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]

        m1s, m2s, _ = curve_fn(y_test, y_score)
        auc = auc_fn(y_test, y_score)
        if method == 'pr':
            # m1s is precision and m2s is recall.
            curves[i] = {
                'xs': m2s,
                'ys': m1s,
                'auc': auc
            }
        elif method == 'roc':
            # m1s is FPR and m2s is TPR
            curves[i] = {
                'xs': m1s,
                'ys': m2s,
                'auc': auc
            }
        y_scores += y_score.tolist()
        y_tests += y_test.tolist()

    # Get curve metrics for overall.
    m1s_all, m2s_all, _ = curve_fn(y_tests, y_scores)
    auc_all = auc_fn(y_tests, y_scores)
    if method == 'pr':
        # m1s is recall and m2s is precision.
        f1_best = float('-inf')
        oop_x = 0.0
        oop_y = 0.0
        for prec, rec in zip(m1s_all, m2s_all):
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > f1_best:
                f1_best = f1
                oop_x = rec
                oop_y = prec

        curves['all'] = {
            'xs': m2s_all,
            'ys': m1s_all,
            'auc': auc_all,
            'oop': (oop_x, oop_y)
        }
    elif method == 'roc':
        # m1s is FPR and m2s is TPR
        diff_best = float('-inf')
        oop_x = 0.0
        oop_y = 0.0
        for fpr, tpr in zip(m1s_all, m2s_all):
            diff = tpr - fpr
            if diff > diff_best:
                diff_best = diff
                oop_x = fpr
                oop_y = tpr

        curves['all'] = {
            'xs': m1s_all,
            'ys': m2s_all,
            'auc': auc_all,
            'oop': (oop_x, oop_y)
        }

    return curves


def get_training_statistics(
        clf: sklearn.base.BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        splits: Union[int, list[list, list]] = None) -> dict:
    """Compute confusion matrix for training data with
        cross-validation.

    Args:
        clf: A Scikit-learn classifier.
        X: The input dataset.
        y: The input labels.
        split_feature: The feature used by K-fold for splitting.
        random_state: The random seed.

    Returns:
        The dictionary containing confusion matrix for each fold.

    """
    cv_result = {}
    if splits is None:
        splits = KFold(n_splits=5, shuffle=True).split(X)

    for i, split in enumerate(splits):
        X_train, y_train = X.loc[split[0]], y.loc[split[0]]

        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)

        tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
        cv_result[f'split_{i}'] = {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp}

    return cv_result


def get_similarity_matrix(
        X: pd.DataFrame,
        y: pd.Series) -> np.ndarray:
    """Get the similarity matrix for the input dataset.

    Args:
        X: The input dataset.
        y: The input labels.

    Returns:
        The similarity matrix.

    """
    data = pd.concat([X, y], axis=1)
    sm = distance_matrix(data.to_numpy(), data.to_numpy())

    return sm
