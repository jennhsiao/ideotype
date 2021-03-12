"""Tests for data_process module."""

import os

from ideotype.data_process import read_data, agg_sims
from ideotype import DATA_PATH


def test_read_data():
    """Test read_data."""
    pass


def test_agg_sims():
    """Test agg_sims."""
    yamlfile = os.path.join(DATA_PATH, 'files', 'filepaths.yml')

    (df_sims, df_sites, df_wea,
     df_params, df_all, df_matured) = read_data(yamlfile)
    df = df_all
    groups = ['cvar', 'site']

    # mean
    how = 'mean'
    mx_mean = agg_sims(df, groups, how)
    assert mx_mean.shape == (100, 129)

    how = 'variance'
    mx_variance = agg_sims(df, groups, how)
    assert mx_variance.shape == (100, 129)

    how = 'std'
    mx_std = agg_sims(df, groups, how)
    assert mx_std.shape == (100, 129)
