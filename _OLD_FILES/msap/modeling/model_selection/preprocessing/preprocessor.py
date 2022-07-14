# -*- coding: utf-8 -*-
"""Data preprocessing object.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    - docstring

"""
import pandas as pd

from .scale import standardize, minmax_normalize, robust_normalize
from .impute import knn_impute, iterative_impute, missforest
from .detect_outliers import isolation_forest, local_outlier_factor


class Preprocessor:
    """Class of data preprocessing object.

    Args:
        clf: A classifier with `fit` method. Optional if `skip_fs` is True.
        scale_mode: Specification for a scaling method.
            {'standard',
             'minmax',
             'robust'}, default='standard'.
        impute_mode: Specification for a missing value imputation method.
            {'knn',
             'iterative',
             'missforest'}, default='knn'.
        outlier_mode: Specification for an outlier detection method.
            {'isolation_forest',
             'lof'}, default='isolation_forest'.
        skip_fs: Skip feature selection if True, default=True.

    Attributes:
        TODO
    """

    def __init__(
            self,
            scale_mode='standard',
            impute_mode='knn',
            outlier_mode='isolation_forest'):
        self.scale_mode = scale_mode
        self.impute_mode = impute_mode
        self.outlier_mode = outlier_mode

    def scale(self, X_df):
        """Scale features.

        Args:
            X_df (pd.DataFrame): Input data.

        Returns:
            (pd.DataFrame): Scaled data.

        """
        if self.scale_mode == 'standard':
            X_new_df = standardize(X_df)
        elif self.scale_mode == 'minmax':
            X_new_df = minmax_normalize(X_df)
        elif self.scale_mode == 'robust':
            X_new_df = robust_normalize(X_df)
        else:
            raise ValueError(f"Invalid scaling mode: {self.scale_mode}")

        return X_new_df

    def impute(self, X_df):
        """Impute missing values.

        Args:
            X_df (pd.DataFrame): Input data.

        Returns:
            (pd.DataFrame): Imputed data.

        """
        if self.impute_mode == 'knn':
            X_new_df = knn_impute(X_df)
        elif self.impute_mode == 'iterative':
            X_new_df = iterative_impute(X_df)
        elif self.impute_mode == 'missforest':
            X_new_df = missforest(X_df)
        else:
            raise ValueError(f"Invalid imputation mode: {self.impute_mode}")

        return X_new_df

    def remove_outliers(self, X_df, y_se):
        """Detect outliers.

        Args:
            X_df (pd.DataFrame): Input data.
            y_se (pd.Series): Target data.

        Returns:
            (list): Indices of outliers.

        """
        data = pd.concat([X_df, y_se], axis=1)

        if self.outlier_mode == 'isolation_forest':
            idxs_outlier = isolation_forest(data)
        elif self.outlier_mode == 'lof':
            idxs_outlier = local_outlier_factor(data)
        elif self.outlier_mode == 'none':
            idxs_outlier = []
        else:
            raise ValueError(
                f"Invalid outlier detection mode: {self.outlier_mode}")

        idxs_inlier = [i for i in range(len(X_df))
                       if i not in idxs_outlier]
        X_df = X_df.iloc[idxs_inlier]
        y_se = y_se.iloc[idxs_inlier]

        return X_df, y_se, idxs_outlier

    def preprocess(self, X_df, y_se):
        """Preprocess input data.

        Args:
            X_df (pd.DataFrame): Input data.
            y_se (pd.Series): Target data.

        Returns:
            (pd.DataFrame): Preprocessed data.

        """
        X_df = self.scale(X_df)
        X_df = self.impute(X_df)
        X_df, y_se, idxs_outlier = self.remove_outliers(X_df, y_se)

        # Remove all the constant columns.
        X_df = X_df.loc[:, (X_df != X_df.iloc[0]).any()]

        return X_df, y_se, idxs_outlier
