# -*- coding: utf-8 -*-
"""A one line summary.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Tune the fig parameters for aesthetic.

"""
from typing import List

from mlxtend.feature_selection import SequentialFeatureSelector
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FONT_SIZE_LABEL = 30


def plot_heatmap(
        X: pd.DataFrame,
        title: str = None,
        cmap: str = None,
        path_save: str = None):
    fig, ax = plt.subplots(figsize=(30, 30))

    if cmap is None:
        cmap = plt.cm.bwr
    elif cmap == 'bwr':
        cmap = plt.cm.bwr
    elif cmap == 'Greys':
        cmap = plt.cm.Greys
    else:
        raise ValueError(f"Invalid cmap {cmap}")

    sns.heatmap(
        data=X,
        cmap=cmap,
        square=True)
    ax.set_xticklabels(
        labels=ax.get_xmajorticklabels(),
        fontsize=12)
    ax.set_yticklabels(
        labels=ax.get_ymajorticklabels(),
        fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=20)

    if path_save is not None:
        fig.savefig(path_save)
        plt.close()
    else:
        plt.show()


def plot_embedded_scatter(
        X: pd.DataFrame,
        y: pd.Series,
        title: str = None,
        path_save: str = None):
    fig, ax = plt.subplots(figsize=(15, 15))

    g = sns.scatterplot(
        data=pd.concat([X, y], axis=1),
        x=X.columns[0],
        y=X.columns[1],
        hue=y.name,
        s=50)
    ax.set_xlabel(
        xlabel=ax.get_xlabel(),
        fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(
        ylabel=ax.get_ylabel(),
        fontsize=FONT_SIZE_LABEL)
    plt.setp(g.get_legend().get_texts(), fontsize='20')
    plt.setp(g.get_legend().get_title(), fontsize='20')

    if title is not None:
        ax.set_title(title, fontsize=40, fontweight='bold')

    if path_save is not None:
        fig.savefig(path_save)
        plt.close()
    else:
        plt.show()


def plot_rfe_line(
        sfs: SequentialFeatureSelector,
        title: str = None,
        path_save: str = None):
    fig, ax = plt.subplots(figsize=(15, 15))

    sfs_result = pd.DataFrame(sfs.get_metric_dict()).transpose().reset_index()
    n_features_best = len(sfs.k_feature_idx_)
    xs = sfs_result['index'].tolist()
    ys = sfs_result['avg_score'].tolist()

    sns.lineplot(
        x=xs,
        y=ys)
    plt.xlim(max(xs) + 1, min(xs) - 1)
    plt.xticks(
        [max(xs), min(xs)]
        + list(range(max(xs), min(xs), -int(len(xs) / 5)))
        + [n_features_best])
    plt.vlines(
        n_features_best, plt.ylim()[0], plt.ylim()[1], linestyles='dashed',
        label='Elbow')
    plt.legend()

    if title is not None:
        ax.set_title(title, fontsize=40, fontweight='bold')

    if path_save is not None:
        fig.savefig(path_save)
        plt.close()
    else:
        plt.show()


def plot_curves(
        curves: dict,
        method: str = 'pr',
        path_save: str = None):
    fig, ax = plt.subplots(figsize=(15, 15))

    xlabel = None
    ylabel = None
    title = None
    if method == 'pr':
        xlabel = "Recall"
        ylabel = "Precision"
        title = "Precision-recall Curve"
        ax.axhline(0.5, label='Random', ls='--')
    elif method == 'roc':
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        title = "Receiver Operating Characteristic Curve"
        ax.plot(ax.get_xlim(), ax.get_ylim(), label='Random', ls="--")
    else:
        raise ValueError(f"Invalid method {method}")

    for k, v in curves.items():
        if k == 'all':
            label = 'Overall'
            alpha = 1.0
            ax.plot(*v['oop'], 'r*', ms=10, label="Optimal Operating Point")
        else:
            label = f"Fold {k + 1}"
            alpha = 0.2
        ax.step(
            v['xs'],
            v['ys'],
            label=f"{label} AUC {round(v['auc'], 3)}",
            alpha=alpha)

    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    if title is not None:
        ax.set_title(title, fontsize=40, fontweight='bold')

    if path_save is not None:
        fig.savefig(path_save)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
        cv_result: dict,
        axis_labels: List[str] = None,
        path_save: str = None):
    N_ROUND = 3
    FONT_SIZE = 25
    FONT_FAMILY = 'sans-serif'
    PADDING_LABEL_X = 10
    PADDING_LABEL_Y = 10

    # Create subplots for the confusion matrix and configure a global setup.
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.01, hspace=0.02)
    axes = [[None for _ in range(4)] for _ in range(4)]
    for i in range(16):
        ax = plt.subplot(gs[i])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

        axes[i // 4][i % 4] = ax

    # Fill 2x2 basic confusion matrix stats.
    tps = np.array(
        [cv_result[f'split_{j}']['tp'] for j in range(len(cv_result))])
    fps = np.array(
        [cv_result[f'split_{j}']['fp'] for j in range(len(cv_result))])
    fns = np.array(
        [cv_result[f'split_{j}']['fn'] for j in range(len(cv_result))])
    tns = np.array(
        [cv_result[f'split_{j}']['tn'] for j in range(len(cv_result))])
    for i, j, metric in [
            (0, 0, tps),
            (0, 1, fps),
            (1, 0, fns),
            (1, 1, tns)]:
        axes[i][j].text(
            x=0.5,
            y=0.5,
            s=f"{round(np.mean(metric), N_ROUND)} +/- "
              f"{round(np.std(metric), N_ROUND)}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=FONT_SIZE,
            family=FONT_FAMILY)

        if i == 0:
            axes[i][j].set_xlabel(
                f"GT\n{axis_labels[j]}",
                fontsize=FONT_SIZE,
                fontweight='bold',
                family=FONT_FAMILY,
                labelpad=PADDING_LABEL_X)
            axes[i][j].xaxis.set_label_position('top')
        if j == 0:
            axes[i][j].set_ylabel(
                f"Predicted\n{axis_labels[i]}",
                fontsize=FONT_SIZE,
                fontweight='bold',
                family=FONT_FAMILY,
                rotation='horizontal',
                horizontalalignment='right',
                verticalalignment='center',
                labelpad=PADDING_LABEL_Y)

        if i == j:
            axes[i][j].set_facecolor('lightgreen')
        else:
            axes[i][j].set_facecolor('lightcoral')

        axes[i][j].spines['left'].set_visible(False)
        axes[i][j].spines['right'].set_visible(False)
        axes[i][j].spines['top'].set_visible(False)
        axes[i][j].spines['bottom'].set_visible(False)

    # Fill totals and style spines.
    ps = tps + fns
    ns = fps + tns
    pps = tps + fps
    pns = fns + tns
    for i, j, metric in [
            (0, 2, pps),
            (1, 2, pns),
            (2, 0, ps),
            (2, 1, ns)]:
        axes[i][j].text(
            x=0.5,
            y=0.5,
            s=f"{round(np.mean(metric), N_ROUND)} +/- "
              f"{round(np.std(metric), N_ROUND)}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=FONT_SIZE,
            family=FONT_FAMILY)
        axes[i][j].spines['left'].set_visible(False)
        axes[i][j].spines['right'].set_visible(False)
        axes[i][j].spines['top'].set_visible(False)
        axes[i][j].spines['bottom'].set_visible(False)

    # Label axes of totals.
    axes[0][2].set_xlabel(
        f"Total",
        fontsize=FONT_SIZE,
        fontweight='bold',
        family=FONT_FAMILY,
        labelpad=PADDING_LABEL_X)
    axes[0][2].xaxis.set_label_position('top')
    axes[2][0].set_ylabel(
        f"Total",
        fontsize=FONT_SIZE,
        fontweight='bold',
        family=FONT_FAMILY,
        rotation='horizontal',
        horizontalalignment='right',
        verticalalignment='center',
        labelpad=PADDING_LABEL_Y)

    # Style spines of unused axes.
    for j in [2, 3]:
        axes[2][j].spines['left'].set_visible(False)
        axes[2][j].spines['right'].set_visible(False)
        axes[2][j].spines['top'].set_visible(False)
        axes[2][j].spines['bottom'].set_visible(False)

    # Fill derived metircs.
    precs = tps / (tps + fps)
    npvs = tns / (tns + fns)
    recs = tps / (tps + fns)
    specs = tns / (fps + tns)
    accs = (tns + tps) / (tns + tps + fns + fps)
    f1s = 2 * precs * recs / (precs + recs)
    for i, j, metric, label in [
            (0, 3, precs[np.isfinite(precs)], 'Precision'),
            (1, 3, npvs[np.isfinite(npvs)], 'NPV'),
            (3, 0, recs[np.isfinite(recs)], 'Recall'),
            (3, 1, specs[np.isfinite(specs)], 'Specificity'),
            (3, 2, accs[np.isfinite(accs)], 'Accuracy'),
            (3, 3, f1s[np.isfinite(f1s)], 'F1')]:
        axes[i][j].text(
            x=0.5,
            y=0.5,
            s=f"{round(np.mean(metric), N_ROUND)} +/- "
              f"{round(np.std(metric), N_ROUND)}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=FONT_SIZE,
            family=FONT_FAMILY,
            color=cm.gray(np.mean(metric)))

        if j == 3 and i != 3:
            axes[i][j].set_ylabel(
                label,
                fontsize=FONT_SIZE,
                fontweight='bold',
                family=FONT_FAMILY,
                rotation='horizontal',
                horizontalalignment='left',
                verticalalignment='center',
                labelpad=PADDING_LABEL_Y)
            axes[i][j].yaxis.set_label_position('right')
        if i == 3:
            axes[i][j].set_xlabel(
                label,
                fontweight='bold',
                fontsize=FONT_SIZE,
                family=FONT_FAMILY,
                labelpad=PADDING_LABEL_X)

        axes[i][j].set_facecolor(cm.Blues(np.mean(metric)))

    if path_save is not None:
        fig.savefig(path_save)
        plt.close()
    else:
        plt.show()
