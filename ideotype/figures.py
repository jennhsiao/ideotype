"""Figure functions."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ideotype.data_process import read_data, process_sims


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
        plt.savefig(
            f'/home/disk/eos8/ach315/upscale/figs/'
            f'bars_pheno_{target}_{phenostage_write}.png',
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
