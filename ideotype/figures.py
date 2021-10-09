"""Figure functions."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from palettable.colorbrewer.diverging import PuOr_7

from ideotype.data_process import read_data, process_sims
from ideotype.analysis import rank_top_phenos
from ideotype.init_params import params_sample
from ideotype.utils import fold


def plot_sims_heatmap(df, sim, agg_method, phenos_ranked,
                      cmap, vmins, vmaxs, save=False):
    """
    Plot out simulation heatmaps.

    Parameters
    ----------
    df : pd.DataFrame
        Queried maizsim output dataframe.
    sim : str
        MAIZSIM output you wish to plot.
    agg_method : str

    """
    # Read in sim data
    df_sims, df_sites, df_wea, df_params, df_all, df_matured = read_data(
        '/home/disk/eos8/ach315/ideotype/ideotype/data/files/'
        'filepaths_present.yml')

    # Set up sites and phenotypes
    sites_unsorted = list(set(df_sims.site))
    sites = sites_unsorted.copy()
    sites.sort()
    phenos = list(set(df_sims.cvar))

    # Set up phenostages
    phenostages = [['"Emerged"'],
                   ['"Tasselinit"'],
                   ['"Tasseled"', '"Silked"'],
                   ['"grainFill"']]
    titles = ['Emerged', 'Tassel initiation',
              'Tasseled & Silked', 'Grain Filling']

    # Visualization
    fig = plt.figure(figsize=(20, 18))

    for index in np.arange(4):
        phenostage = phenostages[index]
        mx_sims = process_sims(df, sites, phenos, phenostage, sim, agg_method)
        df_sims = pd.DataFrame(mx_sims).reindex(phenos_ranked)

        ax = fig.add_subplot(2, 2, index+1)
        sns.heatmap(df_sims, cmap=cmap,
                    vmin=vmins[index], vmax=vmaxs[index],
                    cbar_kws={'shrink': 0.5})
        ax.set_title(f'{titles[index]}', fontweight='light', size=14)
        ax.set_xlabel('sites', fontweight='light', size=12)
        ax.set_ylabel('phenotypes', fontweight='light', size=12)
        plt.xticks(fontweight='light', fontsize=10)
        ax.set_yticks(np.arange(0.5, 100.5))
        ax.set_yticklabels(phenos_ranked, fontweight='light',
                           size=5, rotation=0)

    plt.suptitle(f'{sim}', fontweight='light', x=0.5, y=0.93, size=20)
    fig.subplots_adjust(wspace=0.08)
    fig.subplots_adjust(hspace=0.15)

    if save is True:
        plt.savefig(
            f'/home/disk/eos8/ach315/upscale/figs/heatmap_sims_{sim}.png',
            format='png', dpi=800)


def plot_pheno_summary(df, pheno_stage,
                       target, phenos_ranked,
                       color, alpha,
                       target_phenos=None, target_color=None,
                       target_alpha=None, save=False):
    """
    Plot out phenotype summaries fo sim output.

    Parameters
    ----------
    df : pd.DataFrame
    phenos_ranked : list
    pheno_stage : str
    target : str
    target_phenos : list

    """
    # Parameters
    df_grouped = df.groupby(['cvar', 'pheno']).mean().reset_index()
    sim_values = []

    for pheno in phenos_ranked:
        df_bool = df_grouped[
            (df_grouped.pheno == pheno_stage) &
            (df_grouped.cvar == pheno)][target].shape[0]
        if df_bool == 0:
            sim_values.append(np.nan)
        else:
            sim_value = df_grouped[
                (df_grouped.pheno == pheno_stage) &
                (df_grouped.cvar == pheno)][target].values.item()
            sim_values.append(sim_value)

    # Turn top pheno list into string for plotting purposes
    phenos_str = [str(pheno) for pheno in phenos_ranked]

    # Visualization
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(phenos_str, sim_values, width=0.5, color=color, alpha=alpha)
    ax.set_xlim(-2, 101)
    ax.set_ylabel(target, fontweight='light', size=12)
    ax.set_xlabel('phenotype', fontweight='light', size=12)
    ax.set_title(f'{pheno_stage}', fontweight='light')
    plt.xticks(fontweight='light', fontsize=8, rotation=90)
    plt.yticks(fontweight='light', fontsize=10, rotation=0)

    if target_phenos is not None:
        for target_pheno in target_phenos:
            ax.bar(str(target_pheno),
                   sim_values[phenos_ranked.index(target_pheno)],
                   width=0.5, color=target_color, alpha=target_alpha)

    if save is True:
        phenostage_write = pheno_stage.strip('"')
        if target_phenos is None:
            plt.savefig(
                f'/home/disk/eos8/ach315/upscale/figs/'
                f'bars_pheno_{target}_{phenostage_write}.png',
                format='png', dpi=800)
        else:
            plt.savefig(
                f'/home/disk/eos8/ach315/upscale/figs/'
                f'bars_pheno_{target}_{phenostage_write}_'
                f'targetpheno.png',
                format='png', dpi=800)


def plot_site_summary(df, pheno_stage, target, color, alpha, save=False):
    """
    Plot out site summaries fo sim output.

    Parameters
    ----------
    df : pd.DataFrame
    pheno_stage : str

    """
    # Read in sims data
    df_sims, df_sites, df_wea, df_params, df_all, df_matured = read_data(
        '/home/disk/eos8/ach315/ideotype/ideotype/data/files/'
        'filepaths_present.yml')

    # Parameters
    df_grouped = df.groupby(['site', 'pheno']).mean().reset_index()
    sites = [int(site) for site in df_sites.site]
    sim_values = []

    for site in sites:
        df_bool = df_grouped[
            (df_grouped.pheno == pheno_stage) &
            (df_grouped.site == site)][target].shape[0]
        if df_bool == 0:
            sim_values.append(np.nan)
        else:
            sim_value = df_grouped[
                (df_grouped.pheno == pheno_stage) &
                (df_grouped.site == site)][target].values.item()
            sim_values.append(sim_value)

    # Visualization
    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-2, 61)

    ax.bar(df_sites.site, sim_values, width=0.5, color=color, alpha=alpha)
    ax.set_ylabel(target, fontweight='light', size=12)
    ax.set_xlabel('sites', fontweight='light', size=12)
    plt.xticks(fontweight='light', fontsize=8, rotation=90)
    plt.yticks(fontweight='light', fontsize=10, rotation=0)

    if save is True:
        plt.savefig(
            f'/home/disk/eos8/ach315/upscale/figs/bars_site_{target}.png',
            format='png', dpi=800)


def plot_params_heatmap(df_params, top_phenos,
                        n_phenos_toplot=20, fig_w=9, fig_h=6,
                        save=None, save_text=None):
    """
    Plot params heatmap.

    Parameters
    ----------
    df_params : pd.DataFrame
    top_phenos: list
        List of phenos to plot.
    n_phenos_toplot : int
        -1 - plot all top phenos

    """
    # Rank top_phenos
#    top_phenos = rank_top_phenos(run_name, n_pheno, w_yield, w_disp)

    # Determined parameters perturbed and perturb range
    problem, param_values = params_sample('present', 10)
    param_range = dict(zip(problem['names'], problem['bounds']))
    params = problem['names']
    df_params_fold = pd.DataFrame(columns=params)

    # Normalize parameter values
    if n_phenos_toplot == -1:
        n_phenos_toplot = len(top_phenos)

    if n_phenos_toplot > len(top_phenos):
        n_phenos_toplot = len(top_phenos)

    df_highperformance = df_params.iloc[top_phenos[:n_phenos_toplot], :-1]

    for param in params:
        df_params_fold[param] = fold(df_highperformance[param],
                                     param_range[param][0],
                                     param_range[param][1])

    # Visualize
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(df_params_fold.transpose(), cmap=PuOr_7.mpl_colormap)
    ax.set_xticks(np.arange(df_highperformance.shape[0]))
    ax.set_yticks(np.arange(df_highperformance.shape[1]))
    ax.set_xticklabels(list(df_highperformance.index),
                       size=10, fontweight='light')
    ax.set_yticklabels(list(df_highperformance.columns),
                       size=10, fontweight='light')

    for top_pheno in np.arange(n_phenos_toplot):
        for param in range(len(params)):
            ax.text(top_pheno, param,
                    df_params.transpose().loc[params[param],
                                              top_phenos[top_pheno]],
                    ha='center', color='grey', size=7)
    fig.subplots_adjust(left=0.15)

    if save is True:
        plt.savefig(f'/home/disk/eos8/ach315/upscale/figs/'
                    f'heatmap_params_{save_text}.png',
                    format='png', dpi=800)


def plot_rankchange(n_pheno, w_yield, w_disp, future_run,
                    fig_w=12, fig_h=4, save=None):
    """
    Plot rank change.

    Parameters
    ----------
    n_pheno : int
    w_yield : int
    w_disp : int
    future_run : str
    n_phenos_toplot : int
    save : bool

    """
    # Prep ranks
    top_phenos_present = rank_top_phenos('present', n_pheno, w_yield, w_disp)
    top_phenos_future = rank_top_phenos(future_run, n_pheno, w_yield, w_disp)

    rank_diffs = []
    new_ranks = []
    for item, pheno in enumerate(top_phenos_present):
        try:
            new_rank = top_phenos_future.index(pheno)
            new_ranks.append(new_rank)
            rank_diffs.append(item-new_rank)
        except (ValueError):
            new_ranks.append(new_rank)
            rank_diffs.append(np.nan)

    # Visualization
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(1, 1, 1)

    phenos = top_phenos_present[:]

    y1s = []
    y2s = []
    for item, pheno in enumerate(phenos):
        y1s.append(n_pheno-item)
        y2s.append((n_pheno-item) + rank_diffs[item])

        if rank_diffs[item] < 0:
            plt.arrow(item, n_pheno-item, 0, rank_diffs[item],
                      head_width=0.8,
                      length_includes_head=True,
                      head_starts_at_zero=True,
                      color='tab:orange', alpha=0.8)
        elif rank_diffs[item] > 0:
            plt.arrow(item, n_pheno-item, 0, rank_diffs[item],
                      head_width=0.8,
                      length_includes_head=True,
                      color='tab:purple', alpha=0.8)
        elif rank_diffs[item] == 0:
            plt.scatter(item, n_pheno-item, c='grey', alpha=0.8, marker='_')
        else:
            try:
                new_pheno = top_phenos_future[item]
                if new_pheno in top_phenos_present:
                    plt.scatter(item, n_pheno-item, c='grey',
                                alpha=0.8, marker='x')
                else:
                    plt.scatter(item, n_pheno-item, c='grey',
                                s=200, alpha=0.2, marker='o')
                    plt.text(item-0.5, n_pheno-item-1,
                             new_pheno, size=10, fontweight='light')
            except IndexError:
                print('future top ranks less than present day')

    # x-axis
    ax.set_xlim(-1, len(top_phenos_present))
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(top_phenos_present)))
    ax.set_xticklabels(top_phenos_present, fontweight='light',
                       fontsize=10, rotation=90)

    # y-axis
    min_y = min(min(y1s), min(y2s))
    min_y_rounded = round(min_y/5)*5
    plt.ylabel('ranking', fontweight='light', size=14)
    ax.set_ylim(min_y_rounded-1, n_pheno+1)
    ax.set_yticks(np.arange(min_y_rounded, n_pheno+1, 5))
    ax.set_yticklabels(np.arange(0, abs(min_y_rounded)+n_pheno+1, 5)[::-1],
                       fontweight='light')

    # patch
    rect = plt.Rectangle((-1, 0), len(top_phenos_present)+1, n_pheno+1,
                         facecolor='grey', alpha=0.1)
    ax.add_patch(rect)

    # save
    if save is True:
        plt.savefig(f'/home/disk/eos8/ach315/upscale/figs/'
                    f'rankchange_{future_run}_top{n_pheno}'
                    f'_y{w_yield}_d{w_disp}.png',
                    format='png', dpi=800)
