"""Compilation of functions used for data processing."""
import os
import yaml
from itertools import compress
from datetime import datetime

import pandas as pd
import numpy as np

from ideotype.utils import get_filelist
from ideotype import DATA_PATH


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
            'soil_rt', 'mx_rootdept',
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
    df_sims = pd.read_csv(os.path.join(DATA_PATH, 'sims', sims),
                          dtype={'site': str})

    # 2. site & site-years
    df_sites_all = pd.read_csv(os.path.join(DATA_PATH, 'sites', sites),
                               dtype={'site': str})
    df_siteyears = pd.read_csv(os.path.join(DATA_PATH, 'siteyears', siteyears),
                               dtype={'site': str})
    df_sites = df_sites_all[df_sites_all.site.isin(df_siteyears.site)]
    df_sites.reset_index(inplace=True, drop=True)

    # 3. weather
    df_wea = pd.read_csv(os.path.join(DATA_PATH, 'wea', wea),
                         dtype={'site': str})

    # 4. parameter
    df_params = pd.read_csv(os.path.join(DATA_PATH, 'params', params))
    df_params['cvar'] = df_params.index

    # 5. merge all
    df_sims_params = pd.merge(df_sims, df_params, on='cvar')
    df_sims_params_sites = pd.merge(df_sims_params, df_sites, on='site')

    df_all = pd.merge(df_sims_params_sites,
                      df_wea, on=['site', 'year'])

    # 6. data with simulations that reached maturity only
    df_matured = df_all[df_all.note == '"Matured"']

    return(df_sims, df_sites, df_wea, df_params, df_all, df_matured)


def parse_mature(df_all):
    """
    Further parse maturity in sims.

    Parameters
    ----------
    df_all : pd.DataFrame
        df_all processed through read_data()

    Returns
    -------
    df_extended : pd.DataFrame
        Sims that ended without reaching maturity because
        simulations extended out of growing season 11/29 cut off.
    df_stuck : pd.DataFrame
        Sims that ended without reaching maturity because
        simulations took longer than time cut off.

    """
    # Filter out non-matured sims
    df_notmature = df_all[df_all.note != '"Matured"']
    end_dates = df_notmature.date

    # Extended sims
    # 11/29 = jday 333 or 334 depending on leap year or not
    df_select_extended = [int(
        datetime.strptime(
            date, '%m/%d/%Y').strftime('%j')) >= 333 for date in end_dates]
    df_extended = df_notmature[df_select_extended]

    # Stuck sims
    df_select_stuck = [int(
        datetime.strptime(
            date, '%m/%d/%Y').strftime('%j')) < 333 for date in end_dates]
    df_stuck = df_notmature[df_select_stuck]

    return (df_extended, df_stuck)


def agg_sims(df, groups, how, sim):
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
    sim : str
        Model output.

    Returns:
    --------
    np.matrix
        matrix of aggregated data.

    """
    list_groupindex = []
    for group in groups:
        # find set in index, turn to list
        # since order can get messed up when turned into list
        # re-order and turn back to list again
        list_groupindex.append(list(np.sort(list(set(df[group])))))

    lens = []
    for item in range(len(groups)):
        lens.append(len(list_groupindex[item]))

    # create empty matrix to store final aggregated data
    mx_data = np.empty(shape=lens)
    mx_data[:] = np.nan

    if how == 'mean':
        df_grouped = df.groupby(groups).mean()[sim]

    if how == 'variance':
        df_grouped = df.groupby(groups).var()[sim]

    if how == 'std':
        df_grouped = df.groupby(groups).agg(np.std)[sim]

    for count, item in enumerate(list_groupindex[0]):
        # create empty array to store data for each row (cvar) in matrix
        # a_rows length should equal numebr of sites
        a_rows = np.empty(lens[1])
        # populate list with nans
        a_rows[:] = np.nan

        # list of sites
        sitelist_all = list_groupindex[1]
        # list of sites for that particular cvar
        # (some cvars may have missing data)
        sitelist_cvar = list(df_grouped.loc[(item,)].index)

        # select cvar locations for designated site
        a_rows_index = [sitelist_all.index(site) for site in sitelist_cvar]

        # fill in yield data for specified cvar & site
        a_rows[a_rows_index] = df_grouped.loc[(item,)].values
        mx_data[count] = a_rows

    return mx_data


def process_sims(df, sites, phenos, phenostage, sim, agg_method):
    """
    Process sim queries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of queried info.
        Previously processed with script query_sims.py
    sites : list
        List of simulation sites.
    phenos : list
        List of phenotypes.
    phenostage : list of strings
        - 'all'
        - ['"none:']
        - ['"Germinated"']
        - ['"Emerged"']
        - ['"Tasselinit"']
        - ['"Tasseled"']
        - ['"Silkled"']
        - ['"Matured"']
        - [['"Tasseled"', '"Silked"']]
    sim : str
        Model output to process. Must match column in df.
    agg_method : str
        Method to aggregate model output.
        - 'mean'
        - 'max'

    Returns
    -------
    mx_phys : np.array
        Matrix of processed sims.

    """
    # Filter for pheno stage
    if phenostage == 'all':
        df_phenostage = df

    else:
        if len(phenostage) == 1:
            pheno_stage = phenostage[0]
            df_phenostage = df[df.pheno == pheno_stage]
        else:
            cols = df.columns
            df_phenostage = pd.DataFrame(columns=cols)
            for item in np.arange(len(phenostage)):
                pheno_stage = phenostage[item]
                df_phenostage = df_phenostage.append(
                    df[df.pheno == pheno_stage])

            if df_phenostage.dtypes[sim] == 'O':
                # appending data with date-time columns
                # can make dtype of other columns wonky
                # and could turn into 'object'
                # correct for that here
                df_phenostage[sim] = df_phenostage[sim].astype(float)

    # Group phyiology outputs by phenotype & site
    if agg_method == 'mean':
        df_grouped = df_phenostage.groupby(['cvar', 'site']).mean()
    elif agg_method == 'max':
        df_grouped = df_phenostage.groupby(['cvar', 'site']).max()
    else:
        raise ValueError(f'agg method {agg_method} not supported!')

    # Set up empty matrix to store outputs
    mx_phys = np.empty(shape=[len(phenos), len(sites)])
    mx_phys[:] = np.nan

    # Select outputs
    for count_x, pheno in enumerate(phenos):
        for count_y, site in enumerate(sites):
            df_bool = df_grouped.query(
                f'(cvar=={pheno}) & (site=={int(site)})')[str(sim)].shape[0]
            if df_bool == 0:
                mx_phys[count_x, count_y] = np.nan
            else:
                queried_phys = round(
                    df_grouped.query(
                        f'(cvar=={pheno}) '
                        f'& (site=={int(site)})')[str(sim)].item(), 2)
                mx_phys[count_x, count_y] = queried_phys

    return(mx_phys)


def fetch_sim_values(df, pheno_stage, target, phenos_ranked, sites=None):
    """
    Fetch simulation values.

    Parameters
    ----------
    df : pd.DataFrame
    pheno_stage : str
    target : str
    phenos_ranked : list
    sites : list

    """
    # subset sites if needed
    if sites is not None:
        # ** set is an unordered data structure
        # so does not preserve the insertion order
        # when converting set into list,
        # order can get messed up
        sites_all = list(set(df.site))
        sites_all.sort()
        sites_south = sites_all[:30]
        sites_north = sites_all[30:]

        if sites == 'south':
            df_sub = df[df.site.isin(sites_south)]
        elif sites == 'north':
            df_sub = df[df.site.isin(sites_north)]

    else:
        df_sub = df

    # group df
    df_grouped = df_sub.groupby(['cvar', 'pheno']).mean().reset_index()
    sim_values = []

    # fetch sim values
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

    return(sim_values)


def fetch_norm_mean_disp(run_name):
    """
    Fetch normalized yield mean and yield dispersion.

    Parameters
    ----------
    run_name : str

    """
    df_sims, df_sites, df_wea, df_params, df_all, df_matured = read_data(
        os.path.join(DATA_PATH, 'files', f'filepaths_{run_name}.yml'))
    df_extended, df_stuck = parse_mature(df_all)
    df_all.drop(df_stuck.index, inplace=True)

    df = df_all
    groups = ['cvar', 'site']
    sim = 'dm_ear'

    mx_mean = agg_sims(df, groups, 'mean', sim)
    mx_variance = agg_sims(df, groups, 'variance', sim)
    mx_disp = np.divide(mx_variance, mx_mean)
    df_mean = pd.DataFrame(mx_mean)
    df_disp = pd.DataFrame(mx_disp)
    yield_mean = df_mean.mean(axis=1)
    yield_disp = df_disp.mean(axis=1)

#    yield_mean = df_all.groupby('cvar').mean().dm_ear
#    yield_variance = df_all.groupby('cvar').var().dm_ear
#    yield_disp = yield_variance/yield_mean
    yield_mean_norm = (
        yield_mean-yield_mean.min())/(yield_mean.max()-yield_mean.min())
    yield_disp_norm = (
        yield_disp-yield_disp.min())/(yield_disp.max()-yield_disp.min())

    return(yield_mean_norm, yield_disp_norm)


def fetch_mean_disp_diff(run_name_present, run_name_future, phenos):
    """
    Fetch difference in yield mean & yield dispersion.

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

    diffs_yield = []
    diffs_disp = []

    for pheno in phenos:
        diffs_yield.append(
            yield_mean_norm_future[pheno] - yield_mean_norm_present[pheno])
        diffs_disp.append(
            yield_disp_norm_future[pheno] - yield_disp_norm_present[pheno])

    return(diffs_yield, diffs_disp)


def fetch_mean_stability_diff(run_name_present, run_name_future, phenos):
    """
    Fetch difference in yield mean & yield dispersion.

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

    yield_stability_norm_present = 1-yield_disp_norm_present
    yield_stability_norm_future = 1-yield_disp_norm_future

    diffs_yield = []
    diffs_stability = []

    for pheno in phenos:
        diffs_yield.append(
            yield_mean_norm_future[pheno] - yield_mean_norm_present[pheno])
        diffs_stability.append(
            yield_stability_norm_future[pheno] - yield_stability_norm_present[
                pheno])

    return(diffs_yield, diffs_stability)


def fetch_emps(run_name):
    """
    Fetch emergent properties.

    Parameters
    ----------
    run_name : str

    Return
    ------
    df_emps_combined : pd.DataFrame
    df_emps_std : pd.DataFrame

    """
    # Read in data etc.
    df_sims, df_sites, df_wea, df_params, df_all, df_matured = read_data(
        f'/home/disk/eos8/ach315/ideotype/ideotype/data/files/'
        f'filepaths_{run_name}.yml')
    df_extended, df_stuck = parse_mature(df_all)
    df_all.drop(df_stuck.index, inplace=True)

    # set site as int for dataframe emrging purposes
    df_all.site = df_all.site.astype(int)
    df_stuck.site = df_stuck.site.astype(int)
    df_wea.site = df_wea.site.astype(int)

    # Queried maizsim outputs
    df_phys = pd.read_csv(os.path.join(
        DATA_PATH, 'sims', f'sims_{run_name}_phys.csv'))
    df_leaves = pd.read_csv(os.path.join(
        DATA_PATH, 'sims', f'sims_{run_name}_leaves.csv'))
    df_waterdeficit = pd.read_csv(os.path.join(
        DATA_PATH, 'sims', f'sims_{run_name}_waterdeficit.csv'))
    df_phenology = pd.read_csv(
        os.path.join(DATA_PATH, 'sims', f'sims_{run_name}_pheno.csv'))
    start_dates = [int(
        datetime.strptime(
            date, '%m/%d/%Y').strftime(
                '%j')) for date in df_phenology.date_start]
    df_phenology['jday'] = start_dates

    # grain-fill start date & duration
    df_pheno_grouped = df_phenology[
        df_phenology.pheno == '"grainFill"'].groupby(
        ['cvar', 'site']).mean().reset_index()[
            ['cvar', 'site', 'pheno_days', 'jday']]

    # edate
    df_pheno_gdate = df_phenology[
        df_phenology.pheno == '"Emerged"'].groupby(
        ['cvar', 'site']).mean().reset_index()[
            ['cvar', 'site', 'jday']]
    df_pheno_gdate.columns = ['cvar', 'site', 'edate']
    df_pheno_grouped = df_pheno_grouped.merge(
        df_pheno_gdate, how='left', on=['cvar', 'site'])

    # photosynthesis
    df_phys_grouped = df_phys[
        df_phys.pheno == '"grainFill"'].groupby(
        ['cvar', 'site']).mean().reset_index()[
            ['cvar', 'site', 'An', 'gs']]

    # leaf area
    df_leaves_grouped = df_leaves[
        df_leaves.pheno == '"Tasselinit"'].groupby(
        ['cvar', 'site']).mean().reset_index()[
            ['cvar', 'site', 'LA']]

    # water deficit
    df_wdsum = df_waterdeficit[
        df_waterdeficit.pheno == '"Emerged"'].groupby(
        ['cvar', 'site']).mean().reset_index()[
            ['cvar', 'site', 'water_deficit_mean']]

    # merge all
    df_emps = df_pheno_grouped.merge(
        df_phys_grouped, how='left', on=['cvar', 'site'])
    df_emps = df_emps.merge(
        df_leaves_grouped, how='left', on=['cvar', 'site'])
    df_emps = df_emps.merge(
        df_wdsum, how='left', on=['cvar', 'site'])

    # average within each phenotype
    df_emps_combined = df_emps.groupby('cvar').mean()
    df_emps_combined.reset_index(inplace=True)

    # standardize df_emps
    emps = ['pheno_days', 'jday', 'edate',
            'An', 'gs', 'LA', 'water_deficit_mean']
    df_emps_std = pd.DataFrame(columns=emps)
    for item, emp in enumerate(emps):
        emp_values = df_emps_combined[emp]
        emp_values = [abs(emp_value) for emp_value in emp_values]
        emp_values_std = [(emp_value - min(emp_values))/(
            max(emp_values)-min(emp_values)) for emp_value in emp_values]
        df_emps_std[emp] = emp_values_std

    df_emps_std['cvar'] = df_emps_combined.cvar

    return(df_emps_combined, df_emps_std)


def fetch_sens(df_all_present, df_all_f2100, phenogroups):
    """
    Fetch yield sensitivity.

    Parameters
    ----------
    df_all_present : pd.DataFrame
    df_all_f2100 : pd.DataFrame
    phenogroups : list

    Returns
    -------
    sens : list

    """
    # fetch data for strategy
    sens = []
    for item, phenos in enumerate(phenogroups):
        df_s = df_all_present[
            df_all_present.cvar.isin(phenos)].reset_index(drop=True)
        df_s_present = df_s.copy()[[
            'year', 'cvar', 'site', 'dm_ear', 'temp', 'vpd', 'precip']]

        df_s_f2100 = df_all_f2100[
            df_all_f2100.cvar.isin(phenos)].reset_index(drop=True)[
                ['year', 'cvar', 'site', 'dm_ear', 'temp', 'vpd', 'precip']]
        df_s_f2100.site = df_s_f2100.site.astype(int)
        df_s_merged = df_s_present.merge(
            df_s_f2100, how='left', on=['year', 'cvar', 'site'])
        df_s_mergedmean = df_s_merged.groupby('site').mean()

        xs_present = df_s_mergedmean.temp_x.tolist()
        ys_present = df_s_mergedmean.dm_ear_x.tolist()
        xs_f2100 = df_s_mergedmean.temp_y.tolist()
        ys_f2100 = df_s_mergedmean.dm_ear_y.tolist()

        for item in np.arange(df_s_mergedmean.shape[0]):
            x1 = xs_present[item]
            x2 = xs_f2100[item]
            y1 = ys_present[item]
            y2 = ys_f2100[item]
            sen = ((y2-y1)/y1)/(x2-x1)*100
            sens.append(sen)

    return(sens)


def process_clusters(df_clusters, n_clusters,
                     phenos_top20, phenos_improved, phenos_declined,
                     target, predict_threshold):
    """
    Process clustered phenotypes.

    Parameters
    ----------
    df_clusters : pd.DataFrame
    phenos_top20 : list
    phenos_improved : list
    phenos_declined : list
    target : str
        top20
        improved
        declined

    Returns
    -------
    pheno_groups : list

    """
    df_clusters['top20'] = 0
    top20_index = list(df_clusters[df_clusters.cvar.isin(phenos_top20)].index)
    df_clusters.loc[top20_index, 'top20'] = 1

    df_clusters['improved'] = 0
    improved_index = list(df_clusters[
        df_clusters.cvar.isin(phenos_improved)].index)
    df_clusters.loc[improved_index, 'improved'] = 1

    df_clusters['declined'] = 0
    declined_index = list(df_clusters[
        df_clusters.cvar.isin(phenos_declined)].index)
    df_clusters.loc[declined_index, 'declined'] = 1

    targeted_groups = []
    for group in np.arange(n_clusters):
        phenos_target = df_clusters.query(
            f'group=={group}').query(f'{target}==1').shape[0]
        phenos_total = df_clusters.query(f'group=={group}').shape[0]
        if phenos_target/phenos_total >= predict_threshold:
            targeted_groups.append(group)

    # assemble into phenogroups
    pheno_groups = []
    for group in targeted_groups:
        pheno_group = list(df_clusters.query(f'group=={group}').cvar)
        pheno_groups.append(pheno_group)

    # melt phenogroups into phenos list
    pheno_groups_melt = []
    for item in pheno_groups:
        pheno_groups_melt.extend(item)

    return(targeted_groups, pheno_groups)
