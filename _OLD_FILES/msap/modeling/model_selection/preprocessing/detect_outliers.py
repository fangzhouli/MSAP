# -*- coding: utf-8 -*-
"""Outlier detection methods.

Authors:
    Jason Youn - jyoun@ucdavis.edu
    Fangzhou Li - fzli@ucdavis.edu

"""
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def _get_outlier_idxs(indices):
    """Convert integer style outlier / inlier indices to boolean.

    Args:
        indices (list): -1 for outliers and 1 for inliers.

    Returns:
        (list): Indices of outliers.

    """
    return [i for i, x in enumerate(indices) if x != 1]


def isolation_forest(data_df):
    """Detect outliers using the Isolation Forest algorithm.

    Args:
        data_df (pd.DataFrame): Input data.

    Returns:
        (list): False for outliers and True for inliers.

    """

    data_array = data_df.to_numpy()
    clf = IsolationForest(
        n_jobs=-1,
        contamination=0.05)
    clf.fit(data_array)

    return _get_outlier_idxs(clf.predict(data_array).tolist())


def local_outlier_factor(data_df):
    """Detect outliers using the LOF algorithm.

    Args:
        data_df (pd.DataFrame): Input data.

    Returns:
        (list): False for outliers and True for inliers.

    """
    clf = LocalOutlierFactor(
        n_jobs=-1,
        contamination=0.05)

    return _get_outlier_idxs(clf.fit_predict(data_df.to_numpy()).tolist())
