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
    problem, param_values = params_sample(N_sample)

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
    df : np.matrix or pd.dataframe
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
