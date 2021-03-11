"""Misc utility functions."""
import os

import pandas as pd
import numpy as np
import pytz


def fold(val, min, max):
    """
    Transform values normalized between 0-1 back to their regular range.

    Parameters
    ----------
    val : float
        value to be unfolded.
    min: float
        min of value range.
    max: float
        max of value range.

    """
    fold_list = []
    for i in val:
        fold_i = (i-min)/(max - min)
        fold_list.append(fold_i)
    return fold_list


def unfold(val, min, max):
    """
    Transform values normalized between 0-1 back to their regular range.

    Parameters
    ----------
    val : float
        value to be unfolded.
    min: float
        min of value range.
    max: float
        max of value range.

    """
    unfold_list = []
    for i in val:
        unfold_i = i*(max - min) + min
        unfold_list.append(unfold_i)
    return unfold_list


def get_filelist(path):
    """
    Retrieve all files within given file path.

    Including those in subdirectories.

    Parameter
    ---------
    path : String

    """
    # create a list of file and sub directories names in the given directory
    filelist = os.scandir(path)
    allfiles = list()
    # iterate over all the entries
    for entry in filelist:
        # create full path
        fullpath = os.path.join(path, entry)
        # if entry is a directory then get the list of files in this directory
        if os.path.isdir(fullpath):
            allfiles = allfiles + get_filelist(fullpath)
        else:
            allfiles.append(fullpath)
    return allfiles


def CC_VPD(temp, rh):
    """
    Calculate VPD with temperature and RH.

    Based on Clausius-Clapeyron relation.

    Parameter
    ---------
    temp : Float
        Temperature in ˚C.
    rh : Float
        Relative humidity range between 0 & 1 (fraction, not %).

    Returns
    -------
    vpd : Float
        VPD value calculated based on temp & rh.

    """
    # constant parameters
    Tref = 273.15  # reference temperature
#    Es_Tref = 6.11 # saturation vapor pressure at reference temperature (mb)
    Es_Tref = 0.611  # saturation vapor pressure at reference temperature (kPa)
    Lv = 2.5e+06  # latent heat of vaporation (J/kg)
    Rv = 461  # gas constant for moist air (J/kg)

    # transformed temperature inputs
    Tair = temp + Tref

    # Clausius-Clapeyron relation
    es = Es_Tref*np.exp((Lv/Rv)*(1/Tref - 1/Tair))
    e = es*rh
    vpd = es-e

    return(vpd)


def find_zone(site, siteinfo):
    """
    Find time zone for specific sites.

    Parameters
    ----------
    site :
    siteinfo :

    Returns
    -------
    zone:

    """
    from timezonefinder import TimezoneFinder

    lat = float(siteinfo[siteinfo.site == site].lat)
    lon = float(siteinfo[siteinfo.site == site].lon)
    tf = TimezoneFinder()
    zone = tf.timezone_at(lng=lon, lat=lat)
    return zone


def utc_to_local(times, zone):
    """Convert list of utc timestamps into local time."""
    # convert from pd.DatetimeIndex into python datetime format
    times = times.to_pydatetime()
    # setting up the UTC timezone, requires package 'pytz'
    utc = pytz.timezone('UTC')
    local_datetime = list()

    for time in times:
        utctime = utc.localize(time)  # adding UTC timezone to datetime
        localtime = utctime.astimezone(pytz.timezone(zone))
        datetime = pd.to_datetime(localtime)
        local_datetime.append(datetime)

    return local_datetime


def calc_gdd(temps):
    """
    Maize growing season GDD calculation.

    - calculates GDH with base temperature = 8˚C
    - calculated values divided by 24 to correspond to daily values
    - function returns count of point in which gdd exceeds 100
      which can then be used to identify date in which GDD=100 is reached
    - citation: the solar corridor crop system

    """
    gdd = 0
    for count, temp in enumerate(temps):
        if gdd > 100:
            break
        else:
            if temp-8 < 0:
                gdd += 0
            else:
                gdd += (temp-8)/24
    return(count)


def df_agg(df, groups, how):
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


    def df_import(yamlfile):
    # TODO: make yaml file that points to all these .csv files
    # TODO: and update path accordingly
        """
        Read and process relevant data into dataframe.

        Parameters
        ----------
        yaml : str
            path to yaml file that points to relevant data files

        Returns
        -------
        - df_sims
        - df_sites
        - df_wea
        - df_params
        - df_all
        - df_matured

        """
    # 1. maizsim outputs
    df_sims = pd.read_csv('/home/disk/eos8/ach315/upscale/data/sims_6105.csv',
                          dtype={'site': 'str'})

    # 2. site & site-years
    df_sites_all = pd.read_csv('/home/disk/eos8/ach315/upscale/weadata/site_summary.csv',
                               dtype={'site': str})
    siteyears = pd.read_csv('/home/disk/eos8/ach315/upscale/weadata/siteyears_filtered.csv',
                            dtype={'site': str})
    df_sites = df_sites_all[df_sites_all.site.isin(siteyears.site)]
    df_sites.reset_index(inplace=True, drop=True)

    # 3. weather
    df_wea = pd.read_csv('/home/disk/eos8/ach315/upscale/weadata/wea_summary.csv',
                         dtype={'site': 'str'}, index_col=0)
    df_wea.reset_index(inplace=True, drop=True)

    # 4. parameter
    df_params = pd.read_csv('/home/disk/eos8/ach315/upscale/params/param_opt.csv')
    df_params = df_params.drop(['rmax_ltar'], axis=1)
    df_params['cvar'] = df_params.index

    # 5. merge all
    df_sims_params = pd.merge(df_sims, df_params, on='cvar')
    df_sims_params_sites = pd.merge(df_sims_params, df_sites, on='site')
    df_all = pd.merge(df_sims_params_sites, df_wea, on=['site','year'])

    # 6. data with simulations that reached maturity only
    df_matured = df_all[df_all.note == '"Matured"']

    return(df_sims, df_sites, df_wea, df_params, df_all, df_matured)
