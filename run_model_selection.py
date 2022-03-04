# -*- coding: utf-8 -*-
"""Model selection running script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * include reformat.
    * I don;t like preprocessor...
    * Help for clicks

"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import click

from a2h.modeling.configs import (
    GridSearchConfig,
    ModelSelectionConfig)
from a2h.modeling.model_selection.train import train_grid_search_cv, train_cv
from a2h.modeling.model_selection.preprocessing import Preprocessor
from a2h.utils import (
    ClassifierHandler,
    load_X_and_y,
    dump_X_and_y,
    KFold_by_feature)

os.environ["PYTHONWARNINGS"] = (
    "ignore::RuntimeWarning"
)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.DEBUG)


@click.command()
@click.argument(
    'path-input',
    type=click.Path(exists=True))
@click.argument(
    'path-output',
    type=str)
@click.argument(
    'path-data-preprocessed-dir',
    type=str)
@click.argument(
    'feature-label',
    type=str)
@click.option(
    '--feature-kfold',
    default=None)
@click.option(
    '--load-data-preprocessed',
    type=bool,
    default=False)
@click.option(
    '--random-state',
    type=int,
    default=42)
def main(
        path_input,
        path_output,
        path_data_preprocessed_dir,
        feature_label,
        feature_kfold,
        load_data_preprocessed,
        random_state):
    """
    """
    np.random.seed(random_state)

    cfg_model = ModelSelectionConfig

    if load_data_preprocessed is True:
        logging.info(
            "Loading preprocessed data at "
            f"{path_data_preprocessed_dir}")
    else:
        if path_data_preprocessed_dir is None:
            path_data_preprocessed_dir \
                = cfg_model.get_default_path_data_preprocessed_dir()

        logging.info(
            "Generating preprocessed data at "
            f"{path_data_preprocessed_dir}")
        if not os.path.exists(path_data_preprocessed_dir):
            os.mkdir(path_data_preprocessed_dir)

        data = pd.read_csv(path_input)
        if feature_kfold is not None:
            data = data.set_index(feature_kfold)

        X = data.drop([feature_label], axis=1)
        y = data[feature_label]
        for scale_mode, impute_mode, outlier_mode \
                in tqdm(cfg_model.get_all_preprocessing_combinations()):

            filename_data_prep = cfg_model.get_filename_preprocessed_data(
                scale_mode, impute_mode, outlier_mode)
            filename_outliers = cfg_model.get_filename_outliers(
                scale_mode, impute_mode, outlier_mode)

            try:
                preprocessor = Preprocessor(
                    scale_mode,
                    impute_mode,
                    outlier_mode)
                X_prep, y_prep, idxs_outlier = preprocessor.preprocess(X, y)

                dump_X_and_y(
                    X=X_prep
                    if feature_kfold is None else X_prep.reset_index(),
                    y=y_prep
                    if feature_kfold is None else y_prep.reset_index(
                        drop=True),
                    path_output_data=f"{path_data_preprocessed_dir}/"
                    f"{filename_data_prep}")
                np.savetxt(
                    f"{path_data_preprocessed_dir}/{filename_outliers}",
                    idxs_outlier,
                    fmt='%d')
            except Exception:
                pass

    n_total_combinations \
        = len(cfg_model.get_all_preprocessing_combinations()) \
        * len(cfg_model.get_all_classifier_modes())
    logging.info(
        "Starting the model selection pipeline for "
        f"{n_total_combinations} combinations.")

    # Iterate all combinations.
    results = []  # Store all the scores of models.
    failures = []  # Store all combinations of failed models.
    for i, (scale_mode, impute_mode, outlier_mode) in \
            enumerate(tqdm(
                cfg_model.get_all_preprocessing_combinations(),
                desc="Preprocessing Combinations")):

        filename_data_prep = cfg_model.get_filename_preprocessed_data(
            scale_mode, impute_mode, outlier_mode)

        try:
            X, y = load_X_and_y(
                f"{path_data_preprocessed_dir}/{filename_data_prep}",
                col_y=feature_label)
        except Exception as e:
            logging.debug(
                "This preprocessing, "
                f"{(scale_mode, impute_mode, outlier_mode)}, "
                "does not exist for this run.")

            for j, classifier_mode in enumerate(tqdm(
                    cfg_model.get_all_classifier_modes(),
                    desc="Classifiers")):
                failures += [
                    (i * len(cfg_model.get_all_classifier_modes()) + j,
                     (scale_mode, impute_mode, outlier_mode, classifier_mode),
                     e)]
            continue

        # Create KFold based on the specified index. Use default row id if
        #   None.
        splits = KFold_by_feature(X, 5, feature_kfold, random_state)
        if feature_kfold is not None:
            X = X.drop([feature_kfold], axis=1)

        for j, classifier_mode in enumerate(tqdm(
                cfg_model.get_all_classifier_modes(),
                desc="Classifiers")):

            clf = ClassifierHandler(
                classifier_mode, random_state=cfg_model.RNG_SMOTE).clf
            try:
                # Perform grid search and 5-fold CV if hyperparamer tuning is
                #   available.
                if classifier_mode in GridSearchConfig.CLASSIFIER_MODES:
                    result = train_grid_search_cv(
                        clf=clf,
                        X=X,
                        y=y,
                        param_grid=GridSearchConfig.get_config(
                            classifier_mode).get_param_grid(random_state),
                        splits=splits)
                # Perform only 5-fold CV if hyperparamer tuning is not
                #   available.
                else:
                    result = train_cv(
                        clf=clf,
                        X=X,
                        y=y,
                        splits=splits)

                results += [
                    (i * len(cfg_model.get_all_classifier_modes()) + j,
                     (scale_mode, impute_mode, outlier_mode, classifier_mode),
                     result)]

            except Exception as e:
                failures += [
                    (i * len(cfg_model.get_all_classifier_modes()) + j,
                     (scale_mode, impute_mode, outlier_mode, classifier_mode),
                     e)]

    with open(path_output, 'wb') as f:
        pickle.dump((results, failures), f)


if __name__ == '__main__':
    main()
