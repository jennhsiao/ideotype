"""Compilation of functions used for data processing."""
import os
import yaml

import pandas as pd
import numpy as np


def read_sims(files):
    """
    For the given file path of maizsim model output .txt files.

    - 1. slightly resets column names for model output
    - 2. fetches year/site/cvar info from file name
    - 3. reads in last line of model output
    - 4. documents files with abnormal line output length and does not read them
    - 5. compiles year/site/cvar info & last line of model output into pd.DataFrame
    - 6. issues compiled into pd.DataFrame as well
    - 7. returns: df_sims & df_issues

    Parameters
    ----------



    Returns
    -------

    """
    cols = ['year', 'cvar', 'site', 'date', 'jday', 'time',
            'leaves', 'mature_lvs', 'drop_lvs', 'LA', 'LA_dead', 'LAI',
            'RH', 'leaf_WP', 'PFD', 'Solrad',
            'temp_soil', 'temp_air', 'temp_can', 'ET_dmd', 'ET_suply',
            'Pn', 'Pg', 'resp', 'av_gs',
            'LAI_sunlit', 'LAI_shaded',
            'PFD_sunlit', 'PFD_shaded',
            'An_sunlit', 'An_shaded',
            'Ag_sunlit', 'Ag_shaded',
            'gs_sunlit', 'gs_shaded',
            'VPD', 'N', 'N_dmd', 'N_upt', 'N_leaf', 'PCRL',
            'dm_total', 'dm_shoot', 'dm_ear', 'dm_totleaf',
            'dm_dropleaf', 'df_stem', 'df_root',
            'roil_rt', 'mx_rootdept',
            'available_water', 'soluble_c', 'note']

    return cols  # TODO: continue working on this


def import_data(yamlfile):
    """
    Read and process relevant data into dataframe.

    Parameters
    ----------
    yaml : str
        File path for yaml file containing paths for needed data files.

    Returns
    -------
    - df_sims
    - df_sites
    - df_wea
    - df_params
    - df_all
    - df_matured

    """
    from ideotype import DATA_PATH

    with open(yamlfile, 'r') as yfile:
        dict_files = yaml.safe_load(yfile)

    sims = dict_files['sims']
    sites = dict_files['sites']
    siteyears = dict_files['siteyears']
    wea = dict_files['wea']
    params = dict_files['params']

    # 1. maizsim outputs
    df_sims = pd.read_csv(os.path.join(DATA_PATH, 'files', sims),
                          dtype={'site': str})

    # 2. site & site-years
    df_sites_all = pd.read_csv(os.path.join(DATA_PATH, 'files', sites),
                               dtype={'site': str})
    df_siteyears = pd.read_csv(os.path.join(DATA_PATH, 'files', siteyears),
                               dtype={'site': str})
    df_sites = df_sites_all[df_sites_all.site.isin(df_siteyears.site)]
    df_sites.reset_index(inplace=True, drop=True)

    # 3. weather
    df_wea = pd.read_csv(os.path.join(DATA_PATH, 'files', wea),
                         dtype={'site': str}, index_col=0)
    df_wea.reset_index(inplace=True, drop=True)

    # 4. parameter
    df_params = pd.read_csv(os.path.join(DATA_PATH, 'files', params))
    df_params = df_params.drop(['rmax_ltar'], axis=1)
    df_params['cvar'] = df_params.index

    # 5. merge all
    df_sims_params = pd.merge(df_sims, df_params, on='cvar')
    df_sims_params_sites = pd.merge(df_sims_params, df_sites, on='site')
    df_all = pd.merge(df_sims_params_sites,
                      df_wea, on=['site', 'year'])

    # 6. data with simulations that reached maturity only
    df_matured = df_all[df_all.note == '"Matured"']

    return(df_sims, df_sites, df_wea, df_params, df_all, df_matured)


def agg_sims(df, groups, how):
    """
    Aggregate simulation yield output.

    Parameters:
    -----------
    df : pd.DataFrame
    groups : list of pd columns to group by on
        - ['cvar', 'site']
    how : {'mean', 'variance', 'std'}, default 'mean'
        Type of aggregation method to be performed.
        - std: standard deviation

    Returns:
    --------
    np.matrix
        matrix of aggregated data.

    """
    list_groupindex = []
    for group in groups:
        # find set in index, turn to list
        # since order gets messed up when turned into list
        # so re-order and turn back to list again
        list_groupindex.append(list(np.sort(list(set(df[group])))))

    lens = []
    for item in range(len(groups)):
        lens.append(len(list_groupindex[item]))

    # create empty matrix to store final aggregated data
    mx_data = np.empty(shape=lens)
    mx_data[:] = np.nan

    if how == 'mean':
        df_grouped = df.groupby(groups).mean().dm_ear
        for count, item in enumerate(list_groupindex[0]):
            # create empty array to store data for each row in matrix
            a_rows = np.empty(lens[1])
            a_rows[:] = np.nan

            list1 = list_groupindex[1]
            list2 = list(df_grouped.loc[(item,)].index)  # TODO: tricky to fix!
            a_rows_index = [list1.index(item) for item in list2]  # TODO: test this

            a_rows[a_rows_index] = df_grouped.loc[(item,)].values
            mx_data[count] = a_rows

    if how == 'variance':
        df_grouped = df.groupby(groups).var().dm_ear
        for count, index in enumerate(list_groupindex[0]):
            mx_data[count] = df_grouped.loc[(index,)]

    if how == 'std':
        df_grouped = df.groupby(groups).agg(np.std).dm_ear
        for count, index in enumerate(list_groupindex[0]):
            mx_data[count] = df_grouped.loc[(index,)]

    return mx_data
