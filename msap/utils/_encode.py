# -*- coding: utf-8 -*-
"""Data encoding methods.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
import numpy as np
import pandas as pd


def one_hot_encode(
        X_df: pd.DataFrame,
        cols_cat: list[str] = None) -> pd.DataFrame:
    """Encode categorical features with one-hot encoding.

    Args:
        X_df: Input data.
        vat_cat: Categorical feature names. None if no categorical features.

    Returns:
        Encoded data.

    """
    if cols_cat is None:
        return X_df

    def one_hot_encode_column(col):
        col_ohe = pd.get_dummies(col, prefix=col.name, dummy_na=True)
        col_ohe.loc[col_ohe[col.name + '_nan'] == 1, col_ohe.columns[:-1]]\
            = np.nan
        del col_ohe[col.name + '_nan']

        return col_ohe

    X_cat_encoded_lst = []
    X_df[cols_cat].apply(
        lambda col: X_cat_encoded_lst.append(one_hot_encode_column(col)),
        axis=0)

    return pd.concat(
        [X_df.drop(cols_cat, axis=1)] + X_cat_encoded_lst, axis=1)


def binary_encode(
        X_df: pd.DataFrame,
        cols_cat: list[str] = None) -> pd.DataFrame:
    """Encode categorical features with one-hot encoding.

    Args:
        X_df: Input data.
        vat_cat: Categorical feature names. None if no categorical features.

    Returns:
        Encoded data.

    """
    if cols_cat is None:
        return X_df

    def binary_encode_column(col):
        mapped = col.value_counts().index.tolist()
        col_be = col.map({mapped[0]: 1, mapped[1]: 0})
        col_be.name = col_be.name + f'_{mapped[0]}_1_{mapped[1]}_0'

        return col_be

    X_cat_encoded_lst = []
    X_df[cols_cat].apply(
        lambda col: X_cat_encoded_lst.append(binary_encode_column(col)),
        axis=0)

    return pd.concat(
        [X_df.drop(cols_cat, axis=1)] + X_cat_encoded_lst, axis=1)
