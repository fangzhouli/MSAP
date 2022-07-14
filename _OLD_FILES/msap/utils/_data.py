# -*- coding: utf-8 -*-
"""The utility module for data manipulation.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""


from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


def load_X_and_y(
        path_input_data: str,
        col_y: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data and labels from the given data path.

    Args:
        path_input_data (str): The path of a data file.
        col_y (str): The column name for the target feature if given.

    Returns:
        The training data and labels.

    """
    data = pd.read_csv(path_input_data)

    if col_y is not None:
        y = data[col_y]
        bi_nan = y.isnull()  # Boolean index of missing values.

        X = data.drop([col_y], axis=1)[~bi_nan]
        y = y[~bi_nan]
    else:
        X = data
        y = None

    return X, y


def dump_X_and_y(
        X: pd.DataFrame,
        path_output_data: str,
        y: pd.Series = None) -> None:
    """Store training data and labels to the given data path.

    Args:
        X: The training data.
        path_output_data: The path of a data file.
        y: The label data if given.

    Generates:
        An output file at {path_output_data}.

    """
    if y is not None:
        data = pd.concat([X, y], axis=1)
    else:
        data = X

    data.to_csv(path_output_data, index=False)


def KFold_by_feature(
        X: pd.DataFrame,
        n_splits: int = 5,
        feature: str = None,
        random_state: int = None) -> list[np.ndarray, np.ndarray]:
    """K-fold splitting based on a given feature. If not given, it will
    perform the normal K-fold implemented by Scikit-learn.

    Args:
        X: The training dataset.
        n_splits: The number of splits.
        feature: The feature that splitting is based on.
        random_state: The random seed.

    Returns:
        Splits containing row indices of training and validation datasets.

    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if feature is not None:
        splits = []
        for idx_train, idx_val in cv.split(X[feature].unique()):
            X_train = X[X[feature]
                        .isin(X[feature].unique()[idx_train])]
            X_val = X[X[feature]
                      .isin(X[feature].unique()[idx_val])]
            splits += [(X_train.index.to_numpy(), X_val.index.to_numpy())]
    else:
        splits = list(cv.split(X))

    return splits
