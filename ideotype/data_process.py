"""Compilation of functions used for data processing."""
import os
import yaml
from itertools import compress

import pandas as pd
import numpy as np

from ideotype.utils import get_filelist


def read_sims(path):
    """
    Read and condense all maizsim raw output.

    1. Reset column names for model output
    2. Fetch year/site/cvar info from file name
    3. Read in last line of model output
    4. Document files with abnormal line output length and does not read them
    5. Compile year/site/cvar info & last line of model output
    6. Compile issues

    Parameters
    ----------
    files : str
        Root directory of all simulation outputs.

    Returns
    -------
    DataFrame
        df_sims : df with all simulated yield.
        df_issues : df recording failed site-year recordings.

    """
    fpaths = get_filelist(path)

    fpaths_select = [
        (fpath.split('/')[-1].split('_')[0] == 'out1') and
        (fpath.split('/')[-1].split('.')[-1] == 'txt') for fpath in fpaths]
    fpath_sims = list(compress(fpaths, fpaths_select))

    cols = ['year', 'cvar', 'site', 'date', 'jday', 'time',
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
            'dm_dropleaf', 'df_stem', 'df_root',
            'roil_rt', 'mx_rootdept',
            'available_water', 'soluble_c', 'note']

    data_all = []
    issues = []

    for fpath_sim in fpath_sims:
        # extrating basic file info
        year = int(fpath_sim.split('/')[-3])
        site = fpath_sim.split('/')[-1].split('_')[1]
        cvar = int(fpath_sim.split('/')[-1].split('_')[-1].split('.')[0])

        # reading in file and setting up structure
        with open(fpath_sim, 'r') as f:
            f.seek(0, os.SEEK_END)  # move pointer to end of file
            # * f.seek(offset, whence)
            # * Position computed from adding offset to a reference point,
            # * the reference point is selected by the whence argument.
            # * os.SEEK_SET (=0)
            # * os.SEEK_CUR (=1)
            # * os.SEEK_END (=2)

            try:
                # find current position (now at the end of file)
                # and count back a few positions and read forward from there
                f.seek(f.tell() - 3000, os.SEEK_SET)
                # * f.tell() returns an integer giving the file object’s
                # * current position in the file represented as number of bytes
                # * from the beginning of the file when in binary mode
                # * and an opaque number when in text mode.

                for line in f:
                    f_content = f.readlines()

                if len(f_content[-1]) == 523:  # normal character length
                    sim_output = list(f_content[-1].split(','))
                    data = [i.strip() for i in sim_output]
                    data.insert(0, year)
                    data.insert(1, cvar)
                    data.insert(2, site)
                    data_all.append(data)

                else:
                    issues.append(fpath_sim)

            except ValueError:
                issues.append(fpath_sim.split('/')[-1] + str(', value_error'))

    df_sims = pd.DataFrame(data_all, columns=cols)
    df_sims.dm_total = df_sims.dm_total.astype(float)
    df_sims.dm_ear = df_sims.dm_ear.astype(float)
    df_issues = pd.Series(issues, dtype='str')

    return df_sims, df_issues


def read_data(yamlfile):
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

    df_wea_sub = df_wea.loc[:, [
        'site', 'year', 'temp', 'rh', 'precip', 'solrad', 'vpd']]
    df_all = pd.merge(df_sims_params_sites,
                      df_wea_sub, on=['site', 'year'])

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

    if how == 'variance':
        df_grouped = df.groupby(groups).var().dm_ear

    if how == 'std':
        df_grouped = df.groupby(groups).agg(np.std).dm_ear

    for count, item in enumerate(list_groupindex[0]):
        # create empty array to store data for each row in matrix
        a_rows = np.empty(lens[1])
        a_rows[:] = np.nan

        list1 = list_groupindex[1]
        list2 = list(df_grouped.loc[(item,)].index)
        a_rows_index = [list1.index(item) for item in list2]

        a_rows[a_rows_index] = df_grouped.loc[(item,)].values
        mx_data[count] = a_rows

    return mx_data
