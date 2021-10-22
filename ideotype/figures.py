"""Figure functions."""

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from palettable.colorbrewer.diverging import PuOr_7
from palettable.cartocolors.sequential import PurpOr_6
from palettable.colorbrewer.sequential import YlGn_9
from palettable.wesanderson import Mendl_4

from ideotype.data_process import (read_data,
                                   process_sims,
                                   fetch_norm_mean_disp,
                                   fetch_mean_disp_diff)
from ideotype.analysis import rank_top_phenos
from ideotype.init_params import params_sample
from ideotype.utils import fold
from ideotype import DATA_PATH


def plot_sims_heatmap(df, sim, agg_method, phenos_ranked,
                      cmap, vmins=None, vmaxs=None,
                      yfont_size=8, fig_w=20, fig_h=18,
                      save=False):
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
    titles = ['Emerged',
              'Tassel initiation',
              'Tasseled & Silked',
              'Grain Filling']

    # Visualization
    fig = plt.figure(figsize=(fig_w, fig_h))

    for index in np.arange(4):
        phenostage = phenostages[index]
        mx_sims = process_sims(df, sites, phenos, phenostage, sim, agg_method)
        df_sims = pd.DataFrame(mx_sims).reindex(phenos_ranked)

        ax = fig.add_subplot(2, 2, index+1)
        if (vmins is None) & (vmaxs is None):
            sns.heatmap(df_sims, cmap=cmap,
                        cbar_kws={'shrink': 0.5})
        else:
            sns.heatmap(df_sims, cmap=cmap,
                        vmin=vmins[index], vmax=vmaxs[index],
                        cbar_kws={'shrink': 0.5})

        ax.set_title(f'{titles[index]}', fontweight='light', size=14)
        ax.set_xlabel('sites', fontweight='light', size=12)
        ax.set_ylabel('phenotypes', fontweight='light', size=12)
        plt.xticks(fontweight='light', fontsize=10)
        ax.set_yticks(np.arange(0.5, len(phenos_ranked)+0.5))
        ax.set_yticklabels(phenos_ranked, fontweight='light',
                           size=yfont_size, rotation=0)

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


def plot_cspace_rank(phenos_targeted, mx_present, mx_future,
                     df_climate_x_present, df_climate_y_present,
                     df_climate_x_future, df_climate_y_future,
                     climate_x, climate_y):
    """
    Plot out rank in cspace.

    Parameters
    ----------
    phenos_targeted : list
    climate_x : str
    climate_y : str

    """
    fig = plt.figure(figsize=(16, 10))

    for item, pheno in enumerate(phenos_targeted):
        ax = fig.add_subplot(4, 5, item+1)

        # current climate
        ax.scatter(df_climate_x_present.iloc[pheno],
                   df_climate_y_present.iloc[pheno],
                   marker='o', facecolors='none', edgecolors='grey',
                   alpha=0.6, s=60)
        ax.scatter(df_climate_x_present.iloc[pheno],
                   df_climate_y_present.iloc[pheno],
                   c=mx_present[pheno],
                   cmap=PurpOr_6.mpl_colormap.reversed(),
                   vmin=0, vmax=20, alpha=0.4, s=60)

        # future climate
        ax.scatter(df_climate_x_future.iloc[pheno],
                   df_climate_y_future.iloc[pheno],
                   marker='^', facecolor='none', edgecolors='grey',
                   alpha=0.6, s=60)
        ax.scatter(df_climate_x_future.iloc[pheno],
                   df_climate_y_future.iloc[pheno],
                   c=mx_future[pheno], cmap=PurpOr_6.mpl_colormap.reversed(),
                   marker='^', vmin=0, vmax=20, alpha=0.4, s=60)

        ax.set_xlim(8, 35)
        ax.set_ylim(0, 4.1)
        ax.set_xlabel(climate_x, fontweight='light')
        ax.set_ylabel(climate_y, fontweight='light')
        ax.annotate(pheno, (12, 3.2), fontweight='light', size=10)


def plot_cspace_yield(phenos_targeted, df_grouped_present, df_grouped_future,
                      df_climate_x_present, df_climate_y_present,
                      df_climate_x_future, df_climate_y_future,
                      climate_x, climate_y, vmin=80, vmax=250):
    """
    Plot out yield in cspace.

    Parameters
    ----------
    phenos_targeted : list
    climate_x : str
    climate_y : str

    """
    fig = plt.figure(figsize=(16, 10))

    for item, pheno in enumerate(phenos_targeted):
        df_present = df_grouped_present[df_grouped_present.cvar == pheno]
        df_future = df_grouped_future[df_grouped_future.cvar == pheno]

        ax = fig.add_subplot(4, 5, item+1)

        # current climate
        ax.scatter(df_climate_x_present.iloc[pheno],
                   df_climate_y_present.iloc[pheno],
                   marker='o', facecolors='none', edgecolors='grey',
                   alpha=0.6, s=60)
        ax.scatter(df_climate_x_present.iloc[pheno],
                   df_climate_y_present.iloc[pheno],
                   c=df_present.dm_ear,
                   cmap=YlGn_9.mpl_colormap,
                   vmin=vmin, vmax=vmax, alpha=0.6, s=60)

        # future climate
        ax.scatter(df_climate_x_future.iloc[pheno],
                   df_climate_y_future.iloc[pheno],
                   marker='^', facecolor='none', edgecolors='grey',
                   alpha=0.6, s=60)
        ax.scatter(df_climate_x_future.iloc[pheno],
                   df_climate_y_future.iloc[pheno],
                   c=df_future.dm_ear,
                   cmap=YlGn_9.mpl_colormap,
                   marker='^',
                   vmin=vmin, vmax=vmax, alpha=0.6, s=60)

        ax.set_xlim(8, 35)
        ax.set_ylim(0, 4.1)
        ax.set_xlabel(climate_x, fontweight='light')
        ax.set_ylabel(climate_y, fontweight='light')
        ax.annotate(pheno, (12, 3.2), fontweight='light', size=10)


def plot_mean_disp_change(run_name_present, run_name_future, phenos):
    """
    Plot yield mean and yield dispersion change.

    Parameters
    ----------
    run_name_present : str
    run_name_future : str
    phenos : list

    """
    yield_mean_norm_present, yield_disp_norm_present = fetch_norm_mean_disp(
        run_name_present)
    yield_mean_norm_future, yield_disp_norm_future = fetch_norm_mean_disp(
        run_name_future)
    diffs_yield, diffs_disp = fetch_mean_disp_diff(
        run_name_present, run_name_future, phenos)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.scatter(yield_mean_norm_present, yield_disp_norm_present,
               c='slategrey', s=100, alpha=0.2)
    ax.scatter(yield_mean_norm_present[phenos],
               yield_disp_norm_present[phenos],
               c='tab:purple', s=100, alpha=0.4)

    for item, pheno in enumerate(phenos):
        plt.arrow(yield_mean_norm_present[pheno],
                  yield_disp_norm_present[pheno],
                  diffs_yield[item], diffs_disp[item],
                  color='grey', alpha=0.5,
                  head_width=0.01)

    for pheno in phenos:
        ax.annotate(pheno, (yield_mean_norm_present[pheno],
                            yield_disp_norm_present[pheno]), c='grey')

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel('yield mean', fontweight='light', size=14)
    ax.set_ylabel('dispersion index', fontweight='light', size=14)
    ax.set_title('Yield mean and disparsion - averaged over all sites',
                 fontweight='light', size=15)


def plot_sims_yield(run_name, pheno):
    """
    Plot detailed sim plots for yield.

    Parameters
    ----------
    run_name : str
    pheno : int

    """
    df_sims, df_sites, df_wea, df_params, df_all, df_matured = read_data(
        os.path.join(DATA_PATH, 'files', f'filepaths_{run_name}.yml'))

    cols = ['date', 'jday', 'time',
            'leaves', 'mature_lvs', 'drop_lvs', 'LA', 'LA_dead', 'LAI',
            'RH', 'leaf_WP', 'PFD', 'Solrad',
            'temp_soil', 'temp_air', 'temp_can',
            'ET_dmd', 'ET_suply', 'Pn', 'Pg', 'resp', 'av_gs',
            'LAI_sunlit', 'LAI_shaded',
            'PFD_sunlit', 'PFD_shaded',
            'An_sunlit', 'An_shaded',
            'Ag_sunlit', 'Ag_shaded',
            'gs_sunlit', 'gs_shaded',
            'VPD', 'N', 'N_dmd', 'N_upt', 'N_leaf', 'PCRL',
            'dm_total', 'dm_shoot', 'dm_ear', 'dm_totleaf',
            'dm_dropleaf', 'dm_stem', 'dm_root',
            'soil_rt', 'mx_rootdept',
            'available_water', 'soluble_c', 'note']

    fig = plt.figure(figsize=(20, 50))
    sites = df_sites.site

    for loc in np.arange(60):
        ax = fig.add_subplot(12, 5, loc+1)
        ax.set_ylim(0, 350)
        ax.set_xlim(0, 5000)
        ax.annotate(f'{loc}: {sites[loc]} - {df_sites.iloc[loc]["state"]}',
                    (200, 320))
        site = sites[loc]
        years = df_sims.query(f'cvar=={pheno}').query(f'site=="{site}"').year

        for year in years:
            df = pd.read_csv(
                f'/home/disk/eos8/ach315/upscale/sims/'
                f'{run_name}/{year}/var_{pheno}/'
                f'out1_{site}_{year}_var_{pheno}.txt')
            df.columns = cols
            ax.plot(df.dm_ear, alpha=0.5)
            ax.annotate(year, (len(df), list(df.dm_ear)[-1]), color='grey')


def plot_sims_phenostage(run_name, pheno, df_sims, df_sites, df_phenology):
    """
    Plot phenostage sims for all years for specified pheno.

    Parameters
    ----------
    run_name : str
    pheno : int
    df_sims : pd.DataFrame
        Output from `read_data` function.
    df_sites : pd.DataFrame
        Output from `read_data` function.
    df_phenology : pd.DataFrame
        csv data queried from sim database.

    """
    fig = plt.figure(figsize=(20, 50))
    sites = df_sites.site
    phenostages = ['"Germinated"', '"Emerged"', '"Tasselinit"',
                   '"Tasseled"', '"Silked"', '"grainFill"', '"Matured"']
    colors = ['#66a61e', '#1b9e77',
              Mendl_4.mpl_colors[0],
              Mendl_4.mpl_colors[1],
              Mendl_4.mpl_colors[3],
              Mendl_4.mpl_colors[2]]

    for item, site in enumerate(sites):
        ax = fig.add_subplot(12, 5, item+1)
        ax.set_title(f'{item}: {site} - {df_sites.iloc[item]["state"]}',
                     fontweight='light')
        ax.set_xlim(50, 360)
        ax.set_ylim(1959, 2007)
        jday_months = [32, 61, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        ax.set_xticks(jday_months)
        ax.set_xticklabels([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                           fontweight='light', fontsize=12)

        years = df_sims.query(f'cvar=={pheno}').query(f'site=="{site}"').year
        for year in years[:]:
            df_phenology_sub = df_phenology.query(
                f'cvar=={pheno}').query(f'site=={site}').query(f'year=={year}')

            jdays = []
            for phenostage in phenostages:
                try:
                    jday = df_phenology_sub[
                        df_phenology_sub['pheno'] == phenostage].jday.item()
                    jdays.append(jday)
                except(ValueError):
                    jdays.append(np.nan)

            if jdays[-1] is np.nan:
                # check if pheno reached grain-fill
                # but did not make it to maturity
                df = pd.read_csv(
                    f'/home/disk/eos8/ach315/upscale/sims/'
                    f'{run_name}/{year}/var_{pheno}/'
                    f'out1_{site}_{year}_var_{pheno}.txt')
                df.iloc[-1]['date']
                date = datetime.strptime(df.iloc[-1]['date'], '%m/%d/%Y')
                jday = int(date.strftime('%j'))
                if jday >= 333:
                    # jday 333 = Nov. 29th
                    # set last day of grain-fill to Nov. 29th
                    jdays[-1] = 333

            for loc, color in zip(np.arange(len(phenostages)), colors):
                ax.plot([jdays[loc], jdays[loc+1]], [year, year], color=color)

    fig.subplots_adjust(hspace=0.25)
