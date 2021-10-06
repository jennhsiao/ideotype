"""Compilation of functions for analyzing maizsim output."""

import os
import collections

import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from SALib.analyze import rbd_fast

from ideotype import DATA_PATH
from ideotype.init_params import params_sample
from ideotype.data_process import read_data, process_sims, agg_sims


def run_rbdfast(N_sample, run_name):
    """
    Sensitivity analysis through RBD-FAST method.

    Parameters
    ----------
    N_sample : int
        number of samples to generate.
    run_name : str
        run name for batch sims.

    """
    problem, param_values = params_sample(run_name, N_sample)

    # * param_values cannot directly be used for rbd_fast analysis here
    # since values change with each sampling
    # read in previously saved param.csv file as np.matrix
    fpath_read = os.path.join(os.path.expanduser('~'),
                              'upscale', 'params',
                              f'param_{run_name}.csv'
                              )
    X = genfromtxt(fpath_read, delimiter=',', skip_header=1)

    # TODO: still need code that reads in Y here
    Y = []

    # Calculate sensitivity index
    Si = rbd_fast.analyze(problem, X, Y, print_to_consol=False)

    return Si


def run_pca(df, n):
    """
    Run PCA on dataset.

    Parameters
    ----------
    df : np.matrix or pd.DataFrame
        Data for PCA.
    n : int
        Number of components.

    Returns
    -------
    Dataframe with all PC components.

    """
    x = df
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(x)
    column_labels = [f'PC{comp+1}' for comp in np.arange(n)]
    df_pca = pd.DataFrame(data=principalComponents,
                          columns=column_labels)

    return pca, df_pca


def linear_mod(df, features, target, test_size=0.33):
    """
    Linear model that operates from DF based on features & target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to draw data for linear model.
    features : list
        List of features as strings used to construct linear model.
    target : list
        List of target as string.
    test_size : float
        Default as 0.33 - 33% of data set aside for testing.

    """
    X = df[features]
    y = df[target]

    mod = LinearRegression(fit_intercept=True)
    mod.fit(X, y)
    y_pred = mod.predict(X)
    coefs = mod.coef_
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    return coefs, mse, r2


def identify_top_phenos(run_name, n_pheno=5, w_yield=1, w_disp=1):
    """
    Identify top performing phenotypes.

    Parameters
    ----------
    n_pheno : int
        Number of top phenotypes to identify.
    w_yield : int
        Weight on importance of yield.
        Value between 0 - 1.
    w_disp : int
        Weight on importance of yield dispersion.
        Value between 0 - 1.

    Returns
    -------
    df_pheno : pd.DataFrame
        Dataframe with phenotype performance and site info for mapping.
    mx_pheno : np.array
        Matrix with site, pheno, and phenotype performance info for heatmap.

    """
    df_sims, df_sites, df_wea, df_params, df_all, df_matured = read_data(
        os.path.join(DATA_PATH, 'files', f'filepaths_{run_name}.yml'))

    sites = sorted(list(set(df_all.site)))
    phenos = list(set(df_all.cvar))

    list_top_phenos = [[] for item in np.arange(n_pheno)]

    # Identify high performing combinations
    for site in sites:
        # Filter out data for specific site
        df_sub = df_all.query(f'site=="{site}"')

        # Calculate mean yied and yield dispersion acorss years
        # for specified site
        yield_mean = df_sub.groupby('cvar').mean().dm_ear
        yield_var = df_sub.groupby('cvar').var().dm_ear
        yield_disp = yield_var/yield_mean

        # Standardize yield_mean & yield_disp into between 0 & 1
        yield_mean_norm = (
            yield_mean-yield_mean.min())/(yield_mean.max()-yield_mean.min())
        yield_disp_norm = (
            yield_disp-yield_disp.min())/(yield_disp.max()-yield_disp.min())

        # Identify max yield and min dispersion
        max_yield = yield_mean_norm.max()
        min_disp = yield_disp_norm.min()

        # Calculate distance to theoretical optimal
        dist = [np.sqrt(
            w_yield*(ymean - max_yield)**2 + w_disp*(ydisp - min_disp)**2)
            for ymean, ydisp in zip(yield_mean_norm, yield_disp_norm)]
        df_dist = pd.DataFrame(dist, columns=['dist'])
        top_phenos = list(df_dist.nsmallest(n_pheno, 'dist').index)

        for item in np.arange(len(list_top_phenos)):
            top_pheno = top_phenos[item]
            list_top_phenos[item].append(top_pheno)

    # Set up dataframe with top performing pheno info
    df_pheno = pd.DataFrame(list_top_phenos).transpose()
    df_pheno.columns = [f'pheno{n+1}' for n in np.arange(n_pheno)]
    df_pheno['sites'] = sites
    df_pheno = pd.merge(df_pheno, df_sites, left_on='sites', right_on='site')

    df_sites_sorted = pd.DataFrame(sites)
    df_sites_sorted.columns = ['site']
    df_sites_sorted['site_num'] = np.arange(len(sites))
    df_pheno = pd.merge(
        df_pheno, df_sites_sorted, left_on='sites', right_on='site')

    # Initiate empty matrix
    mx_pheno = np.empty(shape=[len(phenos), len(sites)])
    mx_pheno[:] = np.nan

    # Fill in matrix data
    for item in np.arange(n_pheno):
        mx_pheno[df_pheno[f'pheno{item+1}'], df_pheno['site_num']] = item + 1

    return(df_pheno, mx_pheno)


def top_pheno_prevalence(run_name, n_pheno, intervals):
    """
    Identify the prevalence of top performing phenotypes.

    Parameters
    ----------
    run_name : str
        Simulation run name.
    n_pheno : int
        Number of top phenotypes to identify.
    intervals : int
        Number of intervals to create for yield and dispersion weights.

    Returns
    -------
    df_pheno_prevalence : pd.DataFrame

    """
    pheno_prevalences = []
    list_intervals = [round(item, 2) for item in np.arange(
        0, 1.000001, 1/intervals)]
    w_yields = list_intervals.copy()
    w_disps = list_intervals.copy()
    w_disps.reverse()

    for item in np.arange(intervals):
        df_pheno, mx = identify_top_phenos(
            run_name=run_name,
            n_pheno=n_pheno,
            w_yield=w_yields[item],
            w_disp=w_disps[item])
        # convert matrix with site and ranking info into dataframe
        df = pd.DataFrame(mx)
        # count the number of times each phenotype
        # made it into top rankings (n_pheno) across all locations
        pheno_prevalence = list(df.count(axis=1))
        pheno_prevalences.append(pheno_prevalence)

    df_pheno_prevalence = pd.DataFrame(pheno_prevalences).transpose()

    return(df_pheno_prevalence)


def prevalent_top_pheno(run_name, n_pheno, w_yield, w_disp, site_threshold):
    """
    Identify top performing and prevalent phenotypes.

    Parameters
    ----------
    run_name : str
    site_threshold : int
        Threshold for number of sites phenotype should at least have ranked
        as top performer.

    Returns
    -------
    list_top_pheno : list
        List of top performing prevalent phenotypes.

    """
    df_pheno, mx_pheno = identify_top_phenos(
        run_name, n_pheno, w_yield, w_disp)
    df_prevalence = pd.DataFrame(mx_pheno).notna().astype(int).sum(axis=1)
    df_prevalence_sorted = df_prevalence.sort_values()
    list_top_phenos = df_prevalence_sorted[
        df_prevalence_sorted > site_threshold].index.tolist()
    list_top_phenos.reverse()

    return(list_top_phenos)


def rank_by_yield(df):
    """
    Rank phenotypes by yield only.

    Parameters
    ----------
    df : pd.DataFrame
        MAIZSIM yield output dataframe.
        df_sims or df_mature

    """
    # Prep data
    groups = ['cvar', 'site']
    how = 'mean'
    sim = 'dm_ear'

    mx_mean = agg_sims(df, groups, how, sim)
    df_yield_means = pd.DataFrame(mx_mean)

    # Sort data based on mean yield value
    df_yield_means['mean'] = df_yield_means.mean(axis=1)

    # Rank phenos by yield
    phenos_ranked_by_yield = list(df_yield_means.sort_values(by=['mean'],
                                  axis=0, ascending=False).index)

    return phenos_ranked_by_yield


def rank_all_phenos(run_name, n_pheno, w_yield, w_disp):
    """
    Rank performance for all phenotypes across all locations.

    Parameters
    ----------
    run_name : str
    n_pheno : int
    w_yield : float
    w_disp : float

    Returns
    -------
    phenos_ranked : list

    """
    # Identify ranking for all phenotypes
    df_pheno, mx = identify_top_phenos(
        run_name, n_pheno=n_pheno, w_yield=w_yield, w_disp=w_disp)

    # Rank general performance for all phenotypes across all sites
    performance = []
    for site in np.arange(df_pheno.shape[0]):
        # Select phenotypes ranked by performance from df_pheno
        phenos = df_pheno.iloc[site, :n_pheno].tolist()

        # Assign each phenotype ranked value
        # -- lower values mean better performance)
        pheno_ranks = np.arange(n_pheno)

        # Compile phenotype and ranking info into dict
        dict_rank = dict(zip(phenos, pheno_ranks))

        # Sort dict to order by phenotype
        dict_sorted = collections.OrderedDict(sorted(dict_rank.items()))

        # Append ranking into list of performance
        performance.append(list(dict_sorted.values()))

    # Calculate performance
    # -- phenotypes with lowest sum have best performance overall
    df_rankings = pd.DataFrame(performance).transpose()
    df_performance = df_rankings.sum(axis=1)
    phenos_ranked = list(df_performance.sort_values(ascending=True).index)

    return(df_rankings, phenos_ranked)


def rank_top_phenos(run_name, n_pheno, w_yield, w_disp):
    """
    Rank phenotypes that at least rank top n at sim sites.

    n_pheno : int
        Ranking that phenotype at least should achieve.

    """
    df_pheno, mx = identify_top_phenos('present',
                                       n_pheno=n_pheno,
                                       w_yield=w_yield,
                                       w_disp=w_disp)

    top_phenos = []
    for item in np.arange(n_pheno):
        # Identify phenotypes in each ranking for each site
        top_pheno = list(set(df_pheno.iloc[:, item]))
        top_phenos.extend(top_pheno)
    # Compile all phenotypes
    list_top_phenos = list(set(top_phenos))

    # Determine prevalence of phenotype occurrence
    rank_sums = []
    for item in list_top_phenos:
        rank_list = list(mx[item])
        rank_list_reversed = [(n_pheno + 1) - rank for rank in rank_list]
        rank_sum = np.nansum(rank_list_reversed)
        rank_sums.append(rank_sum)

    df_ranksum = pd.DataFrame({'pheno': list_top_phenos,
                               'rank_sum': rank_sums})
    top_pheno_ranks = list(df_ranksum.sort_values(
        'rank_sum', ascending=False)['pheno'])

    return(top_pheno_ranks)


def phenostage_climate(df_all, df_gseason_climate,
                       df_waterdeficit, phenostage_num):
    """
    Process climate data to get in-season summaries.

    Parameters
    ----------
    df_all : pd.DataFrame
    df_gseason_climate : pd.DataFrame
    df_waterdeficit : pd.DataFrame
    phenostage_num : int
        0 - Emerged
        1 - Tasselinit
        2 - Tasseled & Silked
        3 - Grainfill

    """
    phenostages = [['"Emerged"'], ['"Tasselinit"'],
                   ['"Tasseled"', '"Silked"'], ['"grainFill"']]
    phenos = np.arange(100)
    sites = sites = sorted(list(set(df_all.site)))
    phenostage = phenostages[phenostage_num]

    # temp
    df = df_gseason_climate
    sim = 'temp_air'
    agg_method = 'mean'
    mx_temp = process_sims(df, sites, phenos, phenostage, sim, agg_method)
    df_temp = pd.DataFrame(mx_temp)

    # vpd
    df = df_gseason_climate
    sim = 'vpd'
    agg_method = 'mean'
    mx_vpd = process_sims(df, sites, phenos, phenostage, sim, agg_method)
    df_vpd = pd.DataFrame(mx_vpd)

    # water deficit
    df = df_waterdeficit
    sim = 'water_deficit_mean'
    agg_method = 'mean'
    phenostage = phenostages[phenostage_num]
    mx_wd = process_sims(df, sites, phenos, phenostage, sim, agg_method)
    df_wd = pd.DataFrame(mx_wd)

    return(df_temp, df_vpd, df_wd)


def calc_target_pheno_perct(df_params, phenos_ranked,
                            target_param, comparison):
    """
    Calculate percent of targeted phenotypes with desired param trait.

    df_params : pd.DataFrame
    phenos_ranked : list
    target_param : str
    comparison : str
        'greater' - select phenos with param value
            greater than average param value.
        'less_than' - select phenos with param values
            less than average param value.

    """
    # Calculate target parameter mean value
    target_param_mean = df_params[target_param][:100].mean()

    # Query phenotypes with parameter values greater than parameter mean
    if comparison == 'greater':
        phenos_targetparam = list(df_params[:100].query(
            f'{target_param} > {target_param_mean}').cvar)
    if comparison == 'less_than':
        phenos_targetparam = list(df_params[:100].query(
            f'{target_param} < {target_param_mean}').cvar)

    # Calculate percent
    phenos = []
    for pheno in phenos_targetparam:
        if pheno in phenos_ranked[50:]:
            phenos.append(pheno)
    perct = len(phenos)/len(phenos_targetparam)

    return(phenos_targetparam, perct)
