# -*- coding: utf-8 -*-
"""Pairwise correlation methods.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd


def get_pairwise_correlation(X_df, y_se, method='pearson'):
    """Get correlation and p-value dataframes.

    Args:
        X_df (pd.DataFrame): Features.
        y_se (pd.Series): Labels.
        method (str): Correlation methods,
            options={'pearson', 'spearman', 'kendall'}

    Returns:
        (pd.DataFrame, pd.DataFrame): Correlations and p-values.

    """
    data_df = pd.concat([X_df, y_se], axis=1)

    def pearson_pval(x, y):
        return pearsonr(x, y)[1]

    def spearman_pval(x, y):
        return spearmanr(x, y)[1]

    def kendall_pval(x, y):
        return kendalltau(x, y)[1]

    corr_df = data_df.corr(method=method)
    p_val_df = data_df.corr(method=eval(f"{method}_pval"))

    return corr_df, p_val_df
