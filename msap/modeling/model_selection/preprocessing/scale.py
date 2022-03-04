# -*- coding: utf-8 -*-
"""Scaling methods.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import pandas as pd


def standardize(X_df):
    """Apply the standardize the input data.

    Args:
        X_df (pd.DataFrame): Input data.

    Returns:
        (pd.DataFrame): Scaled data.

    """
    X_array = X_df.to_numpy()
    scaler = StandardScaler().fit(X_array)

    return pd.DataFrame(
        scaler.transform(X_array),
        index=X_df.index,
        columns=X_df.columns)


def minmax_normalize(X_df):
    """Apply the MinMax normalization the input data.

    Args:
        X_df (pd.DataFrame): Input data.

    Returns:
        (pd.DataFrame): Scaled data.

    """
    X_array = X_df.to_numpy()
    scaler = MinMaxScaler().fit(X_array)

    return pd.DataFrame(
        scaler.transform(X_array),
        index=X_df.index,
        columns=X_df.columns)


def robust_normalize(X_df):
    """Apply the robust normalization the input data.

    Args:
        X_df (pd.DataFrame): Input data.

    Returns:
        (pd.DataFrame): Scaled data.

    """
    X_array = X_df.to_numpy()
    scaler = RobustScaler().fit(X_array)

    return pd.DataFrame(
        scaler.transform(X_array),
        index=X_df.index,
        columns=X_df.columns)
