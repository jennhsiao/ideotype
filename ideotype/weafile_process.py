"""Read in hourly weather file."""

import os
import glob
import yaml
from datetime import datetime
from dateutil import tz

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder

from ideotype import DATA_PATH
from ideotype.utils import CC_RH
from ideotype.utils import CC_VPD
from ideotype.nass_process import summarize_nass


def read_wea(year_start, year_end):
    """
    Read in raw hourly weather data.

    - Data source: NOAA Integrated Surface Hourly Database
    - Link: https://www.ncdc.noaa.gov/isd
    - Weather data: temperature, RH, precipitation
    - Raw data stored: ~/data/ISH/
    - Output csv files stored: ~/upscale/weadata/process/

    * note:
    For years 1991-2010, only select data from class 1
    (refer to NSRDB manual p.7-8 for more details)
    - class 1: have complete period of record of 1991-2010.
    - class 2: have complete period of record but with
      significant periods of interpolated, filler,
      or otherwise low-quality input data for solar models.
    - class 3: have have some gaps in the period of record
      but have at least 3 years of data.

    Parameters
    ----------
    year_start : int
    year_end : int

    """
    # setting up np.read_fwf arguments
    colnames = ['time',
                'temp', 'temp_quality',
                'dew_temp', 'dtemp_quality',
                'precip', 'precip_time',
                'precip_depth', 'precip_quality',
                'precip_perhr', 'rh']
    colspecs = [(15, 25),  # time
                (87, 92),  # temp
                (92, 93),  # temp_quality
                (93, 98),  # dew_temp
                (98, 99),  # dtemp_quality
                (105, 8193)]  # precip string

    # Read in relevant file paths
    fpaths_wea = os.path.join(DATA_PATH, 'files', 'filepaths_wea.yml')
    with open(fpaths_wea) as pfile:
        dict_fpaths = yaml.safe_load(pfile)

    # Read in info on conversion between WBAN & USAF id numbering system
    fpath_id_conversion = os.path.join(DATA_PATH,
                                       *dict_fpaths['id_conversion'])
    df_stations = pd.read_csv(fpath_id_conversion, header=None, dtype=str)
    df_stations.columns = ['WBAN', 'USAF']

    # Set up basepath
    basepath = dict_fpaths['basepath']

    # Set up years
    if year_start == year_end:
        years = [year_start]
    else:
        years = np.arange(year_start, year_end+1)

    # Set up date parser for pandas
    dateparser = lambda dates: [datetime.strptime(d, '%Y%m%d%H') for d in dates]  # noqa

    # Loop through years to read in data
    for year in years:
        print(year)  # track progress

        # Check first if file exists already
        if os.path.isfile(os.path.join(basepath, f'temp_{year}.csv')):
            raise ValueError(f'temp_{year}.csv exists!')

        # Set up default timeline
        season_start = '02-01-'
        season_end = '11-30-'
        times = pd.date_range(f'{season_start + str(year)}',
                              f'{season_end + str(year)} 23:00:00',
                              freq='1H')

        arr_temp_sites = np.zeros(shape=(len(times),))
        arr_rh_sites = np.zeros(shape=(len(times),))
        arr_precip_sites = np.zeros(shape=(len(times),))

        # initiate empty list to store all site ids (USAF)
        siteid_all = []

        # For years 1961-1990
        if year < 1991:
            fnames = glob.glob(
                os.path.join(os.path.expanduser('~'),
                             'data', 'ISH', str(year), '*'))

        # For years 1991-2010
        else:
            # Select class1 weather station sites
            fpath_stations_info = os.path.join(DATA_PATH,
                                               *dict_fpaths['stations_info'])
            df_sites = pd.read_csv(fpath_stations_info)
            sites = df_sites.query(
                'CLASS == 1').reset_index().USAF.astype('str')

            # Select sites within specified year that are class1
            sites_year = glob.glob(
                os.path.join(os.path.expanduser('~'),
                             'data', 'ISH', str(year), '*'))
            sites_year = pd.Series([
                site.split('/')[-1].split('-')[0] for site in sites_year])
            sites_year = sites_year[
                sites_year.isin(sites)].reset_index(drop=True)

            # Drop duplicates in sites_year
            sites_year.drop_duplicates(keep='first', inplace=True)

            fnames = []
            for site in sites_year:
                fname = glob.glob(os.path.join(os.path.expanduser('~'),
                                               'data', 'ISH',
                                               str(year),
                                               f'{site}-*'))
                if len(fname) == 1:
                    fnames.append(fname[0])
                else:
                    print(f'choose from files: {fname}')
                    fname = glob.glob(os.path.join(os.path.expanduser('~'),
                                                   'data', 'ISH',
                                                   str(year),
                                                   f'{site}-99999-*'))
                    fnames.append(fname[0])

        for name in fnames:
            # site_id
            siteid_usaf = name.split('/')[-1].split('-')[0]
            siteid_wban = name.split('/')[-1].split('-')[1]

            if siteid_usaf == '999999':
                siteid_usaf = df_stations.query(
                    f'WBAN == "{siteid_wban}"').USAF.item()

            siteid_all.append(siteid_usaf)

            # Read in fixed width weather data
            df = pd.read_fwf(name,
                             names=colnames,
                             colspecs=colspecs,
                             header=None,
                             index_col='time',
                             encoding='latin_1',
                             dtype={'temp': int, 'precip': str},
                             parse_dates=['time'],
                             date_parser=dateparser)

            # Remove duplicated hours, keeping only first occurrence
            # keep = 'first': marks duplicate as True
            # except for first occurrence
            # ~: not selecting for True ends up selecting
            # for the non-duplicated indexes
            # *** note: can't just use df.index.drop_duplicates() since
            # * that only returns a list of the non-duplicated index
            # * but you can't just use that to select non-duplicated rows
            # * since it will also pick up the duplicated rows
            df = df[~df.index.duplicated(keep='first')]

            # Add in missing time values
            # Correct for leap years
            # Filter only for growing season
            df = df.reindex(times, fill_value=np.nan)

            # Find precip data
            df.precip_time = df[
                df['precip'].str.find('ADDAA1') != -1]['precip'].str.split(
                    'ADDAA1').str.get(1).str.slice(0, 2).astype(float)
            df.precip_depth = df[
                df['precip'].str.find('ADDAA1') != -1]['precip'].str.split(
                    'ADDAA1').str.get(1).str.slice(2, 6).astype(float)
            df.precip_quality = df[
                df['precip'].str.find('ADDAA1') != -1]['precip'].str.split(
                    'ADDAA1').str.get(1).str.slice(7, 8)

            # Filter out weather data based on quality code (data manual p.26)
            # Masking unqualified data with NANs:
            # code 3 (Erroneous) &
            # code 7 (Erroneous, data originated from an NCEI data source)
            # *** temp
            quality_temp = (
                df.temp_quality == '3') | (df.temp_quality == '7')
            rows_temp = df[quality_temp].index
            df.loc[rows_temp, 'temp'] = np.nan
            # *** dew temp
            quality_dtemp = (
                df.dtemp_quality == '3') | (df.dtemp_quality == '7')
            rows_dtemp = df[quality_dtemp].index
            df.loc[rows_dtemp, 'dew_temp'] = np.nan
            # *** precip
            quality_precip = (
                df.precip_quality == '3') | (df.precip_quality == '7')
            rows_precip = df[quality_precip].index
            df.loc[rows_precip, 'precip'] = np.nan

            # Replace missing data with NaN
            df.temp = df.temp.replace({9999: np.nan})
            df.dew_temp = df.dew_temp.replace({9999: np.nan})
            df.precip_time = df.precip_time.replace({99: np.nan})
            df.precip_depth = df.precip_depth.replace({9999: np.nan})

            # Calculate hourly precip depth
            df.precip_perhr = df.precip_depth/df.precip_time

            # Account for cases where precip_hr = 0
            # which produces infinite precip_perhr
            df.precip_perhr = df.precip_perhr.replace({np.inf: np.nan})

            # Unit conversion
            df.temp = np.round(df.temp/10, 2)
            df.dew_temp = np.round(df.dew_temp/10, 2)
            df.precip_perhr = np.round(df.precip_perhr/10, 1)

            # calculating RH through Clausius Clapeyron
            df.rh = CC_RH(df.temp, df.dew_temp)
            if df[df.rh > 100].rh.sum() > 100:
                print('rh > 100: ', year, name)

            arr_temp_sites = np.vstack([arr_temp_sites, df.temp])
            arr_rh_sites = np.vstack([arr_rh_sites, df.rh])
            arr_precip_sites = np.vstack([arr_precip_sites, df.precip_perhr])

        # Convert all data for single year into pd.DataFrame
        df_temp_sites = pd.DataFrame(arr_temp_sites.transpose(), index=times)
        df_temp_sites.drop(df_temp_sites.columns[0], axis=1, inplace=True)
        df_temp_sites.columns = siteid_all
        df_temp_sites.sort_index(axis=1, inplace=True)

        df_rh_sites = pd.DataFrame(arr_rh_sites.transpose(), index=times)
        df_rh_sites.drop(df_rh_sites.columns[0], axis=1, inplace=True)
        df_rh_sites.columns = siteid_all
        df_rh_sites.sort_index(axis=1, inplace=True)

        df_precip_sites = pd.DataFrame(
            arr_precip_sites.transpose(), index=times)
        df_precip_sites.drop(df_precip_sites.columns[0], axis=1, inplace=True)
        df_precip_sites.columns = siteid_all
        df_precip_sites.sort_index(axis=1, inplace=True)

        # Output data for each year
        df_temp_sites.to_csv(os.path.join(basepath, f'temp_{year}.csv'))
        df_rh_sites.to_csv(os.path.join(basepath, f'rh_{year}.csv'))
        df_precip_sites.to_csv(os.path.join(basepath, f'precip_{year}.csv'))


def read_solrad(year_start, year_end):
    """
    Read in raw hourly solar radiation data.

    - Data source: NSRDB
    - Source: https://nsrdb.nrel.gov/about/u-s-data.html
    - METSTAT Glo (Wh/m2):
      Total amount of direct and diffuse solar radiation (METSTAT-modeled)
      received on a horizontal surface during the 60-minute period
      ending at the timestamp (refer to NSRDB data manla p.15 Table 3)
    - Raw data stored: ~/data/ISH_NSRD/
    - Output csv files stored: ~/upscale/weadata/process/

    * note:
    For years 1991-2010, only select data from class 1
    (refer to NSRDB manual p.7-8 for more details)
    - class 1: have complete period of record of 1991-2010.
    - class 2: have complete period of record but with
      significant periods of interpolated, filler,
      or otherwise low-quality input data for solar models.
    - class 3: have have some gaps in the period of record
      but have at least 3 years of data.

    Parameters
    ----------
    year_start : int
    year_end : int

    """
    # Read in relevant file paths
    fpaths_wea = os.path.join(DATA_PATH, 'files', 'filepaths_wea.yml')
    with open(fpaths_wea) as pfile:
        dict_fpaths = yaml.safe_load(pfile)

    # Set up basepath
    basepath = dict_fpaths['basepath']

    # Read in info on conversion between WBAN & USAF id numbering system
    fpath_id_conversion = os.path.join(DATA_PATH,
                                       *dict_fpaths['id_conversion'])
    df_stations = pd.read_csv(fpath_id_conversion, header=None, dtype=str)
    df_stations.columns = ['WBAN', 'USAF']
    stations_usaf = df_stations.USAF

    # Set up years
    if year_start == year_end:
        years = [year_start]
    else:
        years = np.arange(year_start, year_end+1)

    # Dataframe setup for years 1961-1990
    colnames = ['year', 'month', 'day', 'hour', 'solrad']
    colspecs = [(1, 3), (4, 6), (7, 9), (10, 12), (23, 27)]

    # Loop through years to read in data
    for year in years:
        print(year)  # track progress

        # Check first if file exists already
        if os.path.isfile(os.path.join(basepath, f'solrad_{year}.csv')):
            raise ValueError(f'solrad_{year}.csv exists!')

        # Set up default timeline
        season_start = '02-01-'
        season_end = '11-30-'
        datetimes_season = pd.date_range(
            f'{season_start + str(year)}',
            f'{season_end + str(year)} 23:00:00', freq='1H')

        # Initiate empty array to store data
        arr_solrad_sites = np.zeros(shape=len(datetimes_season),)

        # initiate empty list to store all site ids (USAF)
        siteid_all = []

        # For years 1961-1990
        if year < 1991:
            # Fetch all file names within year
            fnames = glob.glob(
                os.path.join(os.path.expanduser('~'),
                             'data', 'ISH_NSRD', str(year), '*'))

            for name in fnames:
                siteid_wban = name.split('/')[-1].split('_')[0]
                siteid_usaf = df_stations.query(
                    f'WBAN == "{siteid_wban}"').USAF.item()

                siteid_all.append(siteid_usaf)

                # Read in fixed-width data
                df = pd.read_fwf(name,
                                 skiprows=[0],
                                 header=None,
                                 names=colnames,
                                 colspecs=colspecs)

                # Structure date-time info
                datetimes = df.apply(lambda row: datetime(
                    year, row['month'], row['day'], row['hour']-1), axis=1)

                # Fetch solrad - Global Horizontal Radiation (Wh/m2)
                df_solrad = pd.DataFrame(df.solrad)
                df_solrad.index = datetimes

                # Remove duplicated hours, keeping only first occurrence
                # keep = 'first': marks duplicate as True
                # except for first occurrence
                # ~: not selecting for True ends up selecting
                # for the non-duplicated indexes
                df_solrad = df_solrad[
                    ~df_solrad.index.duplicated(keep='first')]

                # Add in missing time values
                # Correct for leap years
                # Filter only for growing season
                df_solrad = df_solrad.reindex(datetimes_season,
                                              fill_value=np.nan)

                # Replace missing data with NaN
                df_solrad.replace({9999: np.nan}, inplace=True)

                arr_solrad_sites = np.vstack(
                    [arr_solrad_sites, df_solrad.solrad])

            # Convert all data for single year into pd.DataFrame
            df_solrad_sites = pd.DataFrame(
                arr_solrad_sites.transpose(), index=datetimes_season)
            df_solrad_sites.drop(
                df_solrad_sites.columns[0], axis=1, inplace=True)
            df_solrad_sites.columns = siteid_all
            df_solrad_sites.sort_index(axis=1, inplace=True)

            # Output data for each year
            df_solrad_sites.to_csv(
                os.path.join(basepath, f'solrad_{year}.csv'))

        # For years 1991-2010:
        else:
            for station in stations_usaf:
                # Search for specified year-site data
                fname = glob.glob(os.path.join(
                    os.path.expanduser('~'),
                    'data', 'ISH_NSRD', str(year), f'{station}_*.csv'))

                if len(fname) == 1:
                    # Read in file
                    df = pd.read_csv(fname[0])
                    siteid_all.append(station)

                else:
                    print('multiple files!', fname)

                # Format date-time info
                dates = df['YYYY-MM-DD']
                hours = df['HH:MM (LST)']
                hours = [int(hour.split(':')[0])-1 for hour in hours]
                datetimes = [datetime.strptime(
                    dates[item] + '-' + str(hours[item]),
                    '%Y-%m-%d-%H') for item in np.arange(df.shape[0])]

                # Fetch solrad - Global Horizontal Radiation (Wh/m2)
                df_solrad = pd.DataFrame(df['METSTAT Glo (Wh/m^2)'])
                df_solrad.columns = ['solrad']
                df_solrad.index = datetimes

                # Remove duplicated hours, keeping only first occurrence
                # keep = 'first': marks duplicate as True
                # except for first occurrence
                # ~: not selecting for True ends up selecting
                # for the non-duplicated indexes
                df_solrad = df_solrad[
                    ~df_solrad.index.duplicated(keep='first')]

                # Add in missing time values
                # Correct for leap years
                # Filter only for growing season
                df_solrad = df_solrad.reindex(datetimes_season,
                                              fill_value=np.nan)

                # Replace missing data with NaN
                df_solrad.replace({9999: np.nan}, inplace=True)

                # Stacking all data as arrays to make sure
                # all dimensions are correct
                arr_solrad_sites = np.vstack(
                    [arr_solrad_sites, df_solrad.solrad])

            # Convert all data for single year into pd.DataFrame
            df_solrad_sites = pd.DataFrame(
                arr_solrad_sites.transpose(), index=datetimes_season)
            df_solrad_sites.drop(
                df_solrad_sites.columns[0], axis=1, inplace=True)
            df_solrad_sites.columns = siteid_all
            df_solrad_sites.sort_index(axis=1, inplace=True)

            # Output data for each year
            df_solrad_sites.to_csv(
                os.path.join(basepath, f'solrad_{year}.csv'))


def wea_combine(basepath):
    """
    Combine weather data for all years.

    Parameters
    ----------
    basepath : str
        path where all weather data csv files are stored.

    """
    # Set up loop iterables
    csv_files = ['temp_*.csv', 'rh_*.csv', 'precip_*.csv', 'solrad_*.csv']
    csv_names = ['temp_all.csv', 'rh_all.csv',
                 'precip_all.csv', 'solrad_all.csv']

    for csvs, csv_name in zip(csv_files, csv_names):
        print(csv_name)

        # Check if compiled csv file exists already
        if os.path.isfile(os.path.join(basepath, csv_name)):
            print(f'{csv_name} exists already!')

        # Combine data for all years
        else:
            fnames = glob.glob(os.path.join(basepath, csvs))
            # Read in and concat data from all years
            df_all = pd.concat(
                [pd.read_csv(name, index_col=0) for name in fnames])
            # Order df by column so sites are ascending
            df_all.sort_index(axis=1, inplace=True)
            # Order df by index so time is ordered
            # * note: glob.glob doesn't always grab filenames in
            # the order you might think so better to order
            # in this case, solrad was not ordered by year
            df_all.sort_index(axis=0, inplace=True)
            # Output concatenated and sorted dataframe
            df_all.to_csv(os.path.join(basepath, csv_name))


def wea_preprocess(basepath):
    """
    Process weather data.

    Parameters
    ----------
    basepath: str
        path to access weather data

    Returns
    -------
    df_temp
    df_rh
    df_precip
    df_solrad

    """
    # Read in processed weather data
    df_temp = pd.read_csv(
        os.path.join(basepath, 'temp_all.csv'),
        index_col=0, parse_dates=True)
    df_rh = pd.read_csv(
        os.path.join(basepath, 'rh_all.csv'),
        index_col=0, parse_dates=True)
    df_precip = pd.read_csv(
        os.path.join(basepath, 'precip_all.csv'),
        index_col=0, parse_dates=True)
    df_solrad = pd.read_csv(
        os.path.join(basepath, 'solrad_all.csv'),
        index_col=0, parse_dates=True)

    # Identify overlapping stations (columns) between
    # temp/rh/precip dataset & solrad dataset
    cols1 = df_temp.columns
    cols2 = df_solrad.columns
    sites = list(cols1.intersection(cols2))

    # Filter for overlapping sites only
    df_temp = df_temp.loc[:, sites]
    df_rh = df_rh.loc[:, sites]
    df_precip = df_precip.loc[:, sites]
    df_solrad = df_solrad.loc[:, sites]

    return(df_temp, df_rh, df_precip, df_solrad)


def wea_siteyears(df_temp, df_rh, df_precip, df_solrad,
                  gseason_start, gseason_end, crthr):
    """
    Identify valid site-years that satisfy critical hours for gap-flling.

    Parameters
    ----------
    df_temp : pd.DataFrame
    df_rh : pd.dataFrame
    df_precip : pd.DataFrame
    df_solrad : pd.DataFrame
    gseason_start : int
        Start of growing season (month)
    gseason_end : int
        End of growing season (month)
    crthr : int
        critical hours for gap-filling

    Returns
    -------
    siteyears : list

    """
    # Identify site-years that satisfy critical hours for gap-filling
    dfs = [df_temp, df_rh, df_precip, df_solrad]
    final_list = []
    years = np.arange(1961, 2011)
    sites = list(df_temp.columns)

    for df in dfs:
        siteyears_all = list()

        for year in years:
            # Filter out specific year
            df_year = df[(df.index.year == year) &
                         (df.index.month >= gseason_start) &
                         (df.index.month <= gseason_end)]
            siteyears = list()

            for site in sites:
                # Filter out specific site-year
                df_siteyear = pd.DataFrame(df_year.loc[:, site])

                # 4000: ~55% of the number of rows
                # Used as a threshold to toss out site-years
                # that have too many gaps to fill
                # even if they satisfy the critical hours.
                # This is set since I noticed some sites have data
                # recorded every 3 hrs.
                # Valide data collection method, but I wanted to avoid
                # having to gap-fill throuhout that time period,
                # especially for precipitation.
                lim = 4000

                # Only continue processing if have less than ~55% of NaNs
                if int(df_siteyear.isna().sum()) < lim:
                    # Identify whether data entry is NaN
                    # df.notnull() returns TRUE or FALSE,
                    # astype(int) turns TRUE into 1, and FALSE into 0
                    df_siteyear['present'] = df_siteyear.notnull().astype(int)

                    # Calculate cumulative sum based on whether data is
                    # Nan value (1) or not (0)
                    # If there are consecutive missing data,
                    # the cumulative sum for those two rows will be the same,
                    # and can further be used for grouping purposes
                    # to count the number of consecutive missing rows
                    # within each streak of missing data.
                    df_siteyear['csum'] = df_siteyear.present.cumsum()

                    # Select individual timesteps that have missing data
                    df_siteyear = df_siteyear[
                        df_siteyear.loc[:, site].isnull()]

                    # Count the number of consecutive NaNs
                    nans_list = df_siteyear.groupby('csum')['csum'].count()

                    # Only record site-years that have fewer consecutive NaNs
                    # than the critical value set
                    if nans_list[nans_list > crthr].shape[0] == 0:
                        use_siteyear = str(year) + '_' + str(site)
                        siteyears.append(use_siteyear)

            siteyears_all.extend(siteyears)

        final_list.append(siteyears_all)

    # Assign site-years
    siteyears_temp = final_list[0]
    siteyears_rh = final_list[1]
    siteyears_precip = final_list[2]
    siteyears_solrad = final_list[3]

    # Identify all overlapping site-years
    siteyears = list(
        set(siteyears_temp) &
        set(siteyears_rh) &
        set(siteyears_precip) &
        set(siteyears_solrad))

    return(siteyears)


def wea_filter(siteyears, area_threshold, irri_threshold, yearspersite):
    """
    Filter valid site-years based on location, area & irri.

    - Location: limit to continental US (boundaries -123, -72, 19, 53)
    - Planting area
    - Irrigation area
    - Estimated pdate

    Parameters
    ----------
    siteyears : list
        Output of site-years from wea_preprocess()
    area: int
        Planting area threshold.
    irri: int
        Percent land irrigated.
    yearspersite : int
        Minimum number of years of data for each site.

    """
    # Identify total number of unique sites within valid site-years
    sites = list(set([siteyear.split('_')[1] for siteyear in siteyears]))
    sites.sort()

    # Read in relevant file paths
    fpaths_wea = os.path.join(DATA_PATH, 'files', 'filepaths_wea.yml')
    with open(fpaths_wea) as pfile:
        dict_fpaths = yaml.safe_load(pfile)

    # Read in stations info
    fpath_stations_info = os.path.join(DATA_PATH,
                                       *dict_fpaths['stations_info'])
    df_stations = pd.read_csv(fpath_stations_info, dtype={'USAF': str})
    df_stations.USAF.isin(sites)

    # Summarize nass data to fetch planting area & percent irrigated info
    df_nass = summarize_nass()

    # Continental US site boundaries
    lat_min = 19
    lat_max = 53
    lon_min = -123
    lon_max = -72

    # Initiate empty list
    areas = []
    perct_irris = []
    sites_inbound = []
    sites_outbound = []

    for site in sites:
        # Fetch site lat/lon info
        lat = df_stations.query(f'USAF == "{site}"')['ISH_LAT (dd)'].item()
        lon = df_stations.query(f'USAF == "{site}"')['ISH_LON(dd)'].item()

        # Only include sites within continental US boundaries
        if (lat_min <= lat <= lat_max) & (lon_min <= lon <= lon_max):
            # Append sites within bound
            sites_inbound.append(site)

            # Calculate distance between site & all nass sites
            dist = list(enumerate(
                np.sqrt((lat - df_nass.lat)**2 + (lon - (df_nass.lon))**2)))
            df_dist = pd.DataFrame(dist, columns=['rownum', 'distance'])

            # select the five nearest locations and average for
            # cropping area & irrigation percentage
            rows = list(df_dist.nsmallest(5, 'distance').rownum)
            area = df_nass.iloc[rows].area.mean()
            perct_irri = df_nass.iloc[rows].perct_irri.mean()
            areas.append(area)
            perct_irris.append(perct_irri)

        else:
            sites_outbound.append(site)

    # add planting area & irrigation info for filtering purposes
    df_filter = pd.DataFrame({'site': sites_inbound,
                              'area': areas,
                              'perct_irri': perct_irris})
    sites_filtered = df_filter.query(
        f'(area > {area_threshold}) & (perct_irri < {irri_threshold})').site

    # Turn siteyears into dataframe for easier processing
    siteyear_years = [siteyear.split('_')[0] for siteyear in siteyears]
    siteyear_sites = [siteyear.split('_')[1] for siteyear in siteyears]
    df_siteyears = pd.DataFrame({'year': siteyear_years,
                                 'site': siteyear_sites})

    # Filter siteyears based on area & percent irrigated
    siteyears_filtered = df_siteyears[df_siteyears.site.isin(sites_filtered)]

    # Filter out sites that have less than 10 years of data
    df_count = pd.DataFrame(siteyears_filtered.groupby('site').count())
    sites_discard = list(df_count.query(f'year < {yearspersite}').index)
    siteyears_filtered = siteyears_filtered[
        ~siteyears_filtered.site.isin(sites_discard)]

    return(siteyears_filtered)


def make_weafile(siteyears_filtered,
                 df_temp, df_rh, df_precip, df_solrad,
                 outpath):
    """
    Make individual maizsim weather files.

    * Note on handling time zone issues:
    - ISH data (temp, rh, precip): recorded in UTC time
    - NSRB data (solrad): recorded in local time

    * Process:
    1. Select ISH data (temp, rh, precip) based on UTC time
    2. Convert UTC datetime info into local datetime
    3. Write out local datetime info as the date-time columns in maizsim
    4. Select solrad data based on local datetime

    Parameters
    ----------
    siteyears_filtered : list
        List of valid & filtered siteyears

    """
    # Read in station info
    fpaths_wea = os.path.join(DATA_PATH, 'files', 'filepaths_wea.yml')
    with open(fpaths_wea) as pfile:
        dict_fpaths = yaml.safe_load(pfile)

    # Read in info on conversion between WBAN & USAF id numbering system
    fpath_stations_info = os.path.join(DATA_PATH,
                                       *dict_fpaths['stations_info'])
    df_stations = pd.read_csv(fpath_stations_info, dtype={'USAF': str})

    # Package needed to find timezone
    tf = TimezoneFinder()

    for row in np.arange(siteyears_filtered.shape[0]):
        # year & site
        year = siteyears_filtered.iloc[row]['year']
        site = siteyears_filtered.iloc[row]['site']

        # lat & lon
        lat = df_stations.query(f'USAF == "{site}"')['ISH_LAT (dd)'].item()
        lon = df_stations.query(f'USAF == "{site}"')['ISH_LON(dd)'].item()

        # Find and define timezone
        zone = tf.timezone_at(lng=lon, lat=lat)
        from_zone = tz.gettz('UTC')
        to_zone = tz.gettz(zone)

        # Construct dataframe that will hold all weather data
        col = ['jday', 'date', 'hour',
               'solrad', 'temp', 'precip', 'rh', 'co2']
        df_wea = pd.DataFrame(columns=col)

        # UTC datetimes
        season_start = '02-02-'  # TODO: think about this & gseason_start/end
        season_end = '11-30-'
        timestamps = pd.date_range(f'{season_start + year}',
                                   f'{season_end + year} 23:00:00',
                                   freq='1H')
        # Convert timestamp into datetime object
        datetimes = [tstamp.to_pydatetime() for tstamp in timestamps]

        # Assign datetime object UTC timezone
        datetimes_utc = [
            dt.replace(tzinfo=from_zone) for dt in datetimes]

        # Convert UTC datetime to local datetime
        datetimes_local = [
            dt_utc.astimezone(to_zone) for dt_utc in datetimes_utc]

        # Put together df_wea:
        # 1. Select temp, rh, & precip data based on original timestamps (UTC)
        df_wea.temp = list(df_temp.loc[timestamps, site])
        df_wea.rh = list(df_rh.loc[timestamps, site])
        df_wea.precip = list(df_precip.loc[timestamps, site])
        df_wea.co2 = 400
        # 2. Use converted local datetime as time info in df_wea
        df_wea.jday = [int(datetime.strftime(
            dt_local, '%j')) for dt_local in datetimes_local]
        df_wea.date = [datetime.strftime(
            dt_local, "'%m/%d/%Y'") for dt_local in datetimes_local]
        df_wea.hour = [dt_local.hour for dt_local in datetimes_local]
        # 3. Select solrad data based on converted local datetime
        timestamps_local = [datetime.strftime(
            dt_local, '%Y-%m-%d %H:%M:%S') for dt_local in datetimes_local]
        df_wea.solrad = list(df_solrad.loc[timestamps_local, site])

        # Gap-fill df_wea
        df_wea.interpolate(axis=0, inplace=True)

        # Round values one last time for uniform weather data
        # otherwise interpolated points will end up with long floating nums
        df_wea.round({'solrad': 1, 'temp': 1, 'precip': 1, 'rh': 2})

        # Write out df_wea for each site-year
        wea_txt = os.path.join(outpath, f'{site}_{year}.txt')
        if os.path.exists(wea_txt):
            print(f'{site}_{year}.txt exists!')
        else:
            df_wea.to_csv(wea_txt, sep='\t', index=False)


def wea_summarize(siteyears_filtered,
                  df_temp, df_rh, df_precip, df_solrad,
                  gseason_start, gseason_end):
    """
    Summarize growing season weather data.

    * note: linear interpolation prior to summarizing
    * - mean climate conditions
    * - variability within climate: variance/mean

    Parameters
    ----------
    df_temp : pd.DataFrame
    df_rh : pd.DataFrame
    df_precip : pd.DataFrame
    df_solrad : pd.DataFrame

    Returns
    -------
    df_wea_summary : pd.DataFrame
        Summary weather data info.

    """
    temp_all = [np.nan]*siteyears_filtered.shape[0]
    tempvar_all = [np.nan]*siteyears_filtered.shape[0]
    rh_all = [np.nan]*siteyears_filtered.shape[0]
    rhvar_all = [np.nan]*siteyears_filtered.shape[0]
    precip_all = [np.nan]*siteyears_filtered.shape[0]
    precipvar_all = [np.nan]*siteyears_filtered.shape[0]
    solrad_all = [np.nan]*siteyears_filtered.shape[0]
    solradvar_all = [np.nan]*siteyears_filtered.shape[0]

    for item in np.arange(siteyears_filtered.shape[0]):
        # year & site
        year = siteyears_filtered.iloc[item]['year']
        site = siteyears_filtered.iloc[item]['site']

        # Temperature
        df = df_temp
        temp = list(df[
            (df.index.year == int(year)) &
            (gseason_start <= df.index.month) &
            (df.index.month < gseason_end)][site].interpolate(axis=0))
        temp_mean = round(np.nanmean(temp), 2)
        temp_var = np.nanvar(temp)/temp_mean
        temp_all[item] = temp_mean
        tempvar_all[item] = temp_var

        # RH
        df = df_rh
        rh = list(df[
            (df.index.year == int(year)) &
            (gseason_start <= df.index.month) &
            (df.index.month < gseason_end)][site].interpolate(axis=0))
        rh_mean = round(np.nanmean(rh), 2)
        rh_var = np.nanvar(rh)/rh_mean
        rh_all[item] = rh_mean
        rhvar_all[item] = rh_var

        # Precip
        df = df_precip
        precip = list(df[
            (df.index.year == int(year)) &
            (gseason_start <= df.index.month) &
            (df.index.month < gseason_end)][site].interpolate(axis=0))
        precip_mean = round(sum(precip), 2)
        precip_var = np.nanvar(precip)/precip_mean
        precip_all[item] = precip_mean
        precipvar_all[item] = precip_var

        # Solrad
        df = df_solrad
        solrad = list(df[
            (df.index.year == int(year)) &
            (gseason_start <= df.index.month) &
            (df.index.month < gseason_end)][site].interpolate(axis=0))
        solrad_mean = round(np.nanmean(solrad), 2)
        solrad_var = np.nanvar(solrad)/solrad_mean
        solrad_all[item] = solrad_mean
        solradvar_all[item] = solrad_var

    # Calculate VPD based on temperature & RH
    vpd_all = [
        round(CC_VPD(temp, rh/100), 2) for temp, rh in zip(temp_all, rh_all)]

    # Compile summarized growing season met info into dataframe
    df_wea_all = pd.DataFrame({'temp': temp_all,
                               'temp_var': tempvar_all,
                               'rh': rh_all,
                               'rh_var': rhvar_all,
                               'vpd': vpd_all,
                               'precip': precip_all,
                               'precip_var': precipvar_all,
                               'solrad': solrad_all,
                               'solrad_var': solradvar_all})
    df_siteyears = siteyears_filtered.reset_index(drop=True)
    df_wea_summary = df_siteyears.join(df_wea_all)

    return(df_wea_summary)
