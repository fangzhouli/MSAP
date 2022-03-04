# -*- coding: utf-8 -*-
"""Configuration file for the grid search in model selection pipeline.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""


class DefaultConfig:
    # Note: Grid search will only be applied to the following classifiers.
    CLASSIFIER_MODES = [
        'decisiontreeclassifier',
        'adaboostclassifier',
        'randomforestclassifier',
        'mlpclassifier']

    @classmethod
    def get_config(cls, classifier_mode):
        if classifier_mode == 'decisiontreeclassifier':
            return DefaultDecisionTreeClassifierGridSearchConfig
        elif classifier_mode == 'adaboostclassifier':
            return DefaultAdaBoostClassifierGridSearchConfig
        elif classifier_mode == 'randomforestclassifier':
            return DefaultRandomForestClassifierGridSearchConfig
        elif classifier_mode == 'mlpclassifier':
            return DefaultMLPClassifierGridSearchConfig
        else:
            raise ValueError(f"Invalid classifier mode: {classifier_mode}")


class DefaultClassifierGridSearchConfig:
    CLASSIFIER_MODE = ''

    @classmethod
    def get_param_grid(cls, random_state=None):
        pass

    @classmethod
    def append_classifier_prefix(cls, param_grid):
        param_grid_appended = {}

        for key_old in param_grid.keys():
            key_new = cls.CLASSIFIER_MODE + '__' + key_old
            param_grid_appended[key_new] = param_grid[key_old]

        return param_grid_appended


class DefaultDecisionTreeClassifierGridSearchConfig(
        DefaultClassifierGridSearchConfig):
    CLASSIFIER_MODE = 'decisiontreeclassifier'
    CRITERION_MODES = ['gini', 'entropy']
    SPLITTER_MODES = ['best', 'random']
    MAX_DEPTH_START = 1
    MAX_DEPTH_END = 50
    MAX_DEPTH_INCREMENT = 2
    MIN_SAMPLES_SPLIT_START = 2
    MIN_SAMPLES_SPLIT_END = 11
    MIN_SAMPLES_SPLIT_INCREMENT = 2
    MIN_SAMPLES_LEAF_START = 1
    MIN_SAMPLES_LEAF_END = 10
    MIN_SAMPLES_LEAF_INCREMENT = 2

    @classmethod
    def get_param_grid(cls, random_state=None):
        param_grid = {
            'criterion': cls.CRITERION_MODES,
            'splitter': cls.SPLITTER_MODES,
            'max_depth': list(
                range(cls.MAX_DEPTH_START, cls.MAX_DEPTH_END,
                      cls.MAX_DEPTH_INCREMENT)),
            'min_samples_split': list(
                range(cls.MIN_SAMPLES_SPLIT_START, cls.MIN_SAMPLES_SPLIT_END,
                      cls.MIN_SAMPLES_SPLIT_INCREMENT)),
            'min_samples_leaf': list(
                range(cls.MIN_SAMPLES_LEAF_START, cls.MIN_SAMPLES_LEAF_END,
                      cls.MIN_SAMPLES_LEAF_INCREMENT)),
            'random_state': [random_state]
        }

        return cls.append_classifier_prefix(param_grid)


class DefaultAdaBoostClassifierGridSearchConfig(
        DefaultClassifierGridSearchConfig):
    CLASSIFIER_MODE = 'adaboostclassifier'
    ALGORITHM_MODES = ['SAMME', 'SAMME.R']
    N_ESTIMATORS_START = 25
    N_ESTIMATORS_END = 501
    N_ESTIMATORS_INCREMENT = 25
    LEARNING_RATE_START = 0.1
    LEARNING_RATE_END = 2.1
    LEARNING_RATE_INCREMENT = 0.2

    @classmethod
    def get_param_grid(cls, random_state=None):
        param_grid = {
            'n_estimators': list(
                range(cls.N_ESTIMATORS_START, cls.N_ESTIMATORS_END,
                      cls.N_ESTIMATORS_INCREMENT)),
            'learning_rate': [x / 10 for x in list(
                range(int(cls.LEARNING_RATE_START * 10),
                      int(cls.LEARNING_RATE_END * 10),
                      int(cls.LEARNING_RATE_INCREMENT * 10)))],
            'algorithm': cls.ALGORITHM_MODES,
            'random_state': [random_state]
        }

        return cls.append_classifier_prefix(param_grid)


class DefaultRandomForestClassifierGridSearchConfig(
        DefaultClassifierGridSearchConfig):
    CLASSIFIER_MODE = 'randomforestclassifier'
    CRITERION_MODES = ['gini', 'entropy']
    N_ESTIMATORS_START = 25
    N_ESTIMATORS_END = 501
    N_ESTIMATORS_INCREMENT = 25
    MIN_SAMPLES_SPLIT_START = 2
    MIN_SAMPLES_SPLIT_END = 11
    MIN_SAMPLES_SPLIT_INCREMENT = 2
    MIN_SAMPLES_LEAF_START = 1
    MIN_SAMPLES_LEAF_END = 10
    MIN_SAMPLES_LEAF_INCREMENT = 2

    @classmethod
    def get_param_grid(cls, random_state=None):
        param_grid = {
            'criterion': cls.CRITERION_MODES,
            'n_estimators': list(
                range(cls.N_ESTIMATORS_START, cls.N_ESTIMATORS_END,
                      cls.N_ESTIMATORS_INCREMENT)),
            'min_samples_split': list(
                range(cls.MIN_SAMPLES_SPLIT_START, cls.MIN_SAMPLES_SPLIT_END,
                      cls.MIN_SAMPLES_SPLIT_INCREMENT)),
            'min_samples_leaf': list(
                range(cls.MIN_SAMPLES_LEAF_START, cls.MIN_SAMPLES_LEAF_END,
                      cls.MIN_SAMPLES_LEAF_INCREMENT)),
            'random_state': [random_state]
        }

        return cls.append_classifier_prefix(param_grid)


class DefaultMLPClassifierGridSearchConfig(
        DefaultClassifierGridSearchConfig):
    CLASSIFIER_MODE = 'mlpclassifier'
    num_hidden_layers_start = 1
    num_hidden_layers_end = 6
    num_hidden_layers_increment = 1
    num_hidden_neurons_start = 10
    num_hidden_neurons_end = 101
    num_hidden_neurons_increment = 10

    @classmethod
    def get_param_grid(cls, random_state=None):
        param_grid = {
            'hidden_layer_sizes':
                [(nn,) * nl
                    for nn in range(cls.num_hidden_neurons_start,
                                    cls.num_hidden_neurons_end,
                                    cls.num_hidden_neurons_increment)
                    for nl in range(cls.num_hidden_layers_start,
                                    cls.num_hidden_layers_end,
                                    cls.num_hidden_layers_increment)],
            'random_state': [random_state]
        }

        return cls.append_classifier_prefix(param_grid)
