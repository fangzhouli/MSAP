# -*- coding: utf-8 -*-
"""Missing value imputation methods.

Authors:
    Jason Youn - jyoun@ucdavis.edu
    Fangzhou Li - fzli@ucdavis.edu

"""
import sys
import warnings

# missingpy and sklearn compatibility
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base # noqa
from missingpy import MissForest

# Enable iterative_imputer.
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from sklearn.impute import KNNImputer
import pandas as pd


def knn_impute(data_df):
    """Impute missing values using KNN.

    Args:
        data_df (pd.DataFrame): Data containing missing values.

    Returns:
        data_new_df (pd.DataFrame): Data with missing values imputed.

    """
    data_array = data_df.to_numpy()
    imputer = KNNImputer(
        n_neighbors=5)

    data_new_df = pd.DataFrame(
        imputer.fit_transform(data_array),
        index=data_df.index,
        columns=data_df.columns)

    return data_new_df


def iterative_impute(data_df):
    """Impute missing values using the multivariate imputer that estimates each
    feature from all the others.

    Args:
        data_df (pd.DataFrame): Data containing missing values.

    Returns:
        data_new_df (pd.DataFrame): Data with missing values imputed.

    """
    data_array = data_df.to_numpy()
    imputer = IterativeImputer()

    data_new_df = pd.DataFrame(
        imputer.fit_transform(data_array),
        index=data_df.index,
        columns=data_df.columns)

    return data_new_df


def missforest(data_df):
    """Impute missing values using the MissForest imputer.

    Args:
        data_df (pd.DataFrame): Data containing missing values.
        random_state (int, optional): Seed of the pseudo
            random number generator to use.

    Returns:
        data_new_df (pd.DataFrame): Data with missing values imputed.

    """
    data_new_df = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data_array = data_df.to_numpy()
        imputer = MissForest(n_jobs=-1, verbose=0)

        data_new_df = pd.DataFrame(
            imputer.fit_transform(data_array),
            index=data_df.index,
            columns=data_df.columns)

    return data_new_df
