# -*- coding: utf-8 -*-
"""Configuration file for the model selection pipeline.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
import os
import itertools


class DefaultConfig:
    # Default direcroty paths.
    PATH_ROOT = f"{os.path.abspath(os.path.dirname(__file__))}/../.."
    PATH_PREPROCESSED_DIR = f"{PATH_ROOT}/outputs/preprocessed"

    # Default model selection parameters.
    CLASSIFIER_MODES = [
        'decisiontreeclassifier',
        'gaussiannb',
        'multinomialnb',
        'svc',
        'adaboostclassifier',
        'randomforestclassifier',
        'mlpclassifier']
    SCALING_MODES = [
        'standard',
        'minmax',
        'robust']
    MVI_MODES = [
        'knn',
        'iterative',
        'missforest']
    OUTLIER_MODES = [
        'isolation_forest',
        'lof',
        'none']
    RNG_SMOTE = 42567  # To avoid using the same as user's random_state.

    @classmethod
    def get_all_preprocessing_combinations(cls):
        """Return all possible combinations of preprocessing methods.

        Returns:
            (list): A list of tuples, (scale_mode, impute_mode, outlier_mode).

        """
        return list(itertools.product(
            cls.SCALING_MODES,
            cls.MVI_MODES,
            cls.OUTLIER_MODES))

    @classmethod
    def get_all_classifier_modes(cls):
        """Return all possible combinations of model selection.

        Returns:
            (list): A list of tuples,
                (classifier_mode, scale_mode, impute_mode, outlier_mode).

        """
        return cls.CLASSIFIER_MODES

    @classmethod
    def get_default_path_preprocessed_data_dir(cls):
        return cls.PATH_PREPROCESSED_DIR

    @classmethod
    def get_filename_preprocessed_data(
            cls,
            scale_mode,
            impute_mode,
            outlier_mode):
        return f"{scale_mode}_{impute_mode}_{outlier_mode}.csv"

    @classmethod
    def get_filename_outliers(
            cls,
            scale_mode,
            impute_mode,
            outlier_mode):
        return f"{scale_mode}_{impute_mode}_{outlier_mode}_outlier_indices.txt"
