# -*- coding: utf-8 -*-
"""Model evaluation script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
import click

from a2h.modeling.model_evaluation.statistics import (
    get_embedded_data,
    get_selected_features,
    get_curve_metrics,
    get_training_statistics,
    get_similarity_matrix)
from a2h.explanatory_analysis import get_pairwise_correlation
from a2h.utils import (
    ClassifierHandler,
    load_X_and_y,
    KFold_by_feature)
from a2h.utils.plot import (
    plot_heatmap,
    plot_embedded_scatter,
    plot_rfe_line,
    plot_curves,
    plot_confusion_matrix)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO)

METHODS_PC = ['pearson', 'spearman', 'kendall']
METHODS_EMBEDDING = ['tsne', 'pca']
METHODS_CURVE = ['pr', 'roc']
CLASSIFIER_MODES = [
    'decisiontreeclassifier',
    'gaussiannb',
    'multinomialnb',
    'svc',
    'adaboostclassifier',
    'randomforestclassifier',
    'mlpclassifier']


def parse_model_selection_result(ms_result: tuple) -> list:
    """Parse the model selection result tuple and get the best models.

    Args:
        ms_result: Model selection result tuple.

    Returns:
        List of best model and statistics for each classifiers.

    """
    candidates, _ = ms_result
    candidates = [(i, c, cv['best']) for i, c, cv in candidates]

    f1s_mean = []
    for i, c, cv_best in candidates:
        # Iterate over splits to calculate average F1 score.
        f1s = [cv_best[f'split_{j}']['f1'] for j in range(len(cv_best) - 1)]
        f1s_mean += [np.mean(np.nan_to_num(f1s))]

    candidates = list(zip(candidates, f1s_mean))
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    best_candidate_per_clf = []
    for clf in CLASSIFIER_MODES:
        for (i, c, cv_best), f1_mean in candidates:
            if c[3] == clf:
                if cv_best['param'] is not None:
                    cv_best['param'] = {k.split('__')[-1]: v
                                        for k, v in cv_best['param'].items()}

                best_candidate_per_clf += [((i, c, cv_best), f1_mean)]
                break

    return best_candidate_per_clf


@click.command()
@click.argument(
    'path-input-model-selection-result',
    type=click.Path(exists=True))
@click.argument(
    'path-input-preprocessed-data-dir',
    type=click.Path(exists=True))
@click.argument(
    'path-input-data-raw',
    type=click.Path(exists=True))
@click.argument(
    'path-output-dir',
    type=str)
@click.argument(
    'feature-label',
    type=str)
@click.option(
    '--feature-kfold',
    type=str,
    default=None)
@click.option(
    '--random-state',
    type=int,
    default=42)
def main(
        path_input_model_selection_result,
        path_input_preprocessed_data_dir,
        path_input_data_raw,
        path_output_dir,
        feature_label,
        feature_kfold,
        random_state):
    """
    """
    if not os.path.exists(path_output_dir):
        os.mkdir(path_output_dir)

    model_selection_result = None
    with open(path_input_model_selection_result, 'rb') as f:
        model_selection_result = pickle.load(f)

    best_candidate_per_clf = parse_model_selection_result(
        model_selection_result)
    best_candidate = max(best_candidate_per_clf, key=lambda x: x[1])
    _, best_combination, best_cv_result = best_candidate[0]
    best_scale_mode, best_impute_mode, best_outlier_mode, best_clf \
        = best_combination
    pd.DataFrame(best_candidate_per_clf).to_csv(
        f"{path_output_dir}/best_clfs.csv")

    # X_raw, _ = load_X_and_y(path_input_data_raw, col_y=feature_label)

    X, y = load_X_and_y(
        f"{path_input_preprocessed_data_dir}/"
        f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}.csv",
        col_y=feature_label)
    # idxes_outlier = np.loadtxt(
    #     f"{path_input_preprocessed_data_dir}/"
    #     f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}"
    #     "_outlier_indices.txt",
    #     delimiter='\n',
    #     dtype=int)

    splits = KFold_by_feature(
        X=X,
        n_splits=5,
        feature=feature_kfold,
        random_state=random_state)
    X = X.drop([feature_kfold], axis=1)

    clf = ClassifierHandler(
        classifier_mode=best_clf,
        params=best_cv_result['param'],
        random_state=random_state).clf

    # Plot pairwise correlation heatmaps.
    for method in METHODS_PC:
        corr, pval = get_pairwise_correlation(
            X, y, method=method)
        y_corr = corr[feature_label].drop([feature_label])
        y_pval = pval[feature_label].drop([feature_label])
        idxes_rank = y_corr.abs().argsort().tolist()[::-1]

        rank = pd.concat(
            [y_corr[idxes_rank], y_pval[idxes_rank]],
            axis=1)
        rank.columns = ['corr', 'p-value']
        rank.to_csv(f"{path_output_dir}/pc_rank_{method}.csv")

        plot_heatmap(
            corr,
            title=f"Pairwise {method.capitalize()} Correlation",
            path_save=f"{path_output_dir}/pc_{method}.png")

    # Plot similarity matrix for the data points heatmap.
    sm = get_similarity_matrix(X, y)
    plot_heatmap(
        sm,
        title=f"Similarity Matrix",
        cmap='Greys',
        path_save=f"{path_output_dir}/sim.png")

    # Plot embedded data points.
    y_scatter = y.map({1.0: 'Success', 0.0: 'Fail'})
    y_scatter.name = 'Translation'
    for method in METHODS_EMBEDDING:
        X_embedded = pd.DataFrame(
            get_embedded_data(
                X,
                method=method, random_state=random_state))
        X_embedded.columns = ['First Dimension', 'Second Dimension']
        plot_embedded_scatter(
            X_embedded,
            y_scatter,
            title=f"{method.upper()}",
            path_save=f"{path_output_dir}/embed_{method}.png")

    # Calculate and plot feature selection for the best model.
    sfs = get_selected_features(clf, X, y, splits)
    plot_rfe_line(
        sfs,
        title="Recursive Feature Elimination",
        path_save=f"{path_output_dir}/rfe.png")
    pd.DataFrame(sfs.get_metric_dict()).transpose().reset_index().to_csv(
        f"{path_output_dir}/rfe_result.csv", index=False)

    # Calculate and plot curves, all classifiers and the best model.
    for method in METHODS_CURVE:
        try:
            curve_metrics = get_curve_metrics(
                clf, X, y, method, splits)
        except Exception as e:
            logger.info(
                f"{method} skipped due to data inbalance. Error Type: "
                f"{type(e)}. Error message: {e}")
            continue

        plot_curves(
            curve_metrics,
            method=method,
            path_save=f"{path_output_dir}/{method}.png")

    # # Plot outliers.
    # y_in_out = ['Inlier' for _ in range(len(X_raw))]
    # for idx in idxes_outlier:
    #     y_in_out[idx] = 'Outlier'
    # y_in_out = pd.Series(y_in_out)
    # y_in_out.name = 'Inlier/Outlier'
    # for method in METHODS_EMBEDDING:
    #     X_raw_embedded = pd.DataFrame(
    #         get_embedded_data(
    #             X_raw.drop([feature_kfold], axis=1),
    #             method=method,
    #             random_state=random_state))
    #     X_raw_embedded.columns = ['First Dimension', 'Second Dimension']
    #     plot_embedded_scatter(
    #         X_raw_embedded,
    #         y_in_out,
    #         title=f"Outlier Detection with {method.upper()}",
    #         path_save=f"{path_output_dir}/outliers_{method}.png")

    # Plot confusion matrix with various metrics for validation.
    del best_cv_result['param']
    plot_confusion_matrix(
        cv_result=best_cv_result,
        axis_labels=['Success', 'Failure'],
        path_save=f"{path_output_dir}/cm.png")

    # Plot confusion matrix with various metrics for validation.
    best_cv_result_train = get_training_statistics(
        clf, X, y, splits)
    plot_confusion_matrix(
        cv_result=best_cv_result_train,
        axis_labels=['Success', 'Failure'],
        path_save=f"{path_output_dir}/cm_train.png")


if __name__ == '__main__':
    main()
