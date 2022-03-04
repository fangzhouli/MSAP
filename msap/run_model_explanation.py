# -*- coding: utf-8 -*-
"""Model evaluation script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * docstring.
    * logging.
    * Stop hard coding the classfier, instead input clf pickle file.

"""
import os

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click

from .utils.clf import ClassifierHandler
from .utils.data import load_X_and_y, split_by_feature


@click.command()
@click.argument(
    'path-input-data',
    type=click.Path(exists=True))
@click.argument(
    'path-output-dir',
    type=str)
@click.option(
    '--id-kfold',
    type=str,
    default=None)
@click.option(
    '--random-state',
    type=int,
    default=42)
def main(
        path_input_data,
        path_output_dir,
        id_kfold,
        random_state):
    """
    """
    if not os.path.exists(path_output_dir):
        os.mkdir(path_output_dir)

    X, y = load_X_and_y(
        path_input_data,
        col_y='exact_translation_1_1_0_0')
    if id_kfold is not None:
        X = X.drop([id_kfold], axis=1)

    """Best for 0.0
    ('minmax', 'iterative', 'lof', 'randomforestclassifier'),
    'param': {
        'criterion': 'gini',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 50,
        'random_state': 42
    }
    """
    clf = ClassifierHandler(
        classifier_mode='randomforestclassifier',
        params={
            'criterion': 'gini',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 50,
            'random_state': 42},
        random_state=random_state).clf
    clf.fit(X, y)
    X_resampled, y_resampled = clf[:-1].fit_resample(X, y)

    # Global explanation using SHAP.
    explainer = shap.TreeExplainer(clf.named_steps['randomforestclassifier'])
    shap_values = explainer(X_resampled[:500])

    shap.plots.bar(shap_values[:, :, 1], show=False)
    fig = plt.gcf()
    fig.set_figwidth(30)
    fig.set_figheight(10)
    plt.savefig(f"{path_output_dir}/shap_bar.png")
    plt.savefig(f"{path_output_dir}/shap_bar.svg", format='svg')
    plt.close()

    shap.plots.beeswarm(shap_values[:, :, 1], show=False)
    fig = plt.gcf()
    fig.set_figwidth(30)
    fig.set_figheight(10)
    plt.savefig(f"{path_output_dir}/shap_beeswarm.png")
    plt.savefig(f"{path_output_dir}/shap_beeswarm.svg", format='svg')
    plt.close()

    shap.plots.force(shap_values[:, :, 1], matplotlib=True, show=False)
    fig = plt.gcf()
    fig.set_figwidth(30)
    fig.set_figheight(10)
    plt.savefig(f"{path_output_dir}/shap_force.png")
    plt.savefig(f"{path_output_dir}/shap_force.svg", format='svg')
    plt.close()

    # # False pos.
    # shap.plots.waterfall(shap_values[7, :, 1])

    # # True neg.
    # shap.plots.waterfall(shap_values[14, :, 1])


if __name__ == '__main__':
    main()
