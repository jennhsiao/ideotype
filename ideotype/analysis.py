"""Compilation of functions for analyzing maizsim output."""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from SALib.analyze import rbd_fast


def run_rbdfast(problem, X, Y):
    """
    Sensitivity analysis through RBD-FAST method.

    Parameters
    ----------
    problem : dict
        SALib parameter problem definition.
    X : np.matrix
        parameter inputs.
    Y : np.matrix
        model outputs.

    """
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
