"""Compilation of functions for analyzing maizsim output."""

import os

import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from SALib.analyze import rbd_fast

from ideotype.init_params import params_sample


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


def cluster_sites(df, features, n):
    """
    Cluster simulation sites into climate space.

    Parameters
    ----------
    df : pd.DataFrame
        Data to cluster.
    features : list
        List of features to cluster on.
    n : int
        Number of clusters.

    Returns
    -------

    """
    pass


def identify_top_phenos(df_all, df_sites, n_pheno=5, w_yield=1, w_disp=1):
    """
    Identify top performing phenotypes.

    Parameters
    ----------
    df_all : pd.DataFrame
    df_sites : pd.DataFrame
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
