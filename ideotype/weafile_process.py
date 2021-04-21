"""Read in hourly weather file."""

import os
import glob
import yaml
from datetime import datetime

import numpy as np
import pandas as pd

from ideotype import DATA_PATH


def __dateparse(dates):
    """
    Datetime parse for pd.read_fwf.

    Parameters
    ----------
    dates : list
        Dates to parse.

    """
    [datetime.strptime(date, "%Y%m%d%H") for date in dates]


def read_wea(year_start, year_end):
    """
    Read in raw hourly weather data for years 1961-1990.

    - Data source: NOAA Integrated Surface Hourly Database
    - Link: https://www.ncdc.noaa.gov/isd
    - Weather data: temperature, RH, precipitation

    Parameters
    ----------
    year_start : int
    year_end : int
    Returns
    -------
    - temp_all_6190.csv
    - precip_all_6190.csv
    - rh_all_6190.csv

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

    # Set up years
    if year_start == year_end:
        years = [year_start]
    else:
        years = np.arange(year_start, year_end+1)

    # Set up dateparsing method when reading in climate data
    
    dateparse = lambda dates: [datetime.strptime(date, "%Y%m%d%H") for date in dates]

    # Loop through years to read in data
    for year in years:
        print(year)  # track progress

        # Set up default timeline
        season_start = '03-01-'
        season_end = '11-30-'
        times = pd.date_range(f'{season_start + str(year)}',
                              f'{season_end + str(year)} 23:00:00',
                              freq='1H')

        # For years 1961-1990
        if year < 1991:
            fnames = glob.glob(
                os.path.join(os.path.expanduser('~'),
                             'data', 'ISH', str(year), '*'))

        # For years 1991-2005
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
                                                   f'{site}-99999-*'))[0]
                    fnames.append(fname)

        for name in fnames:
            # WBAN site_id
            site_id = name.split('/')[-1].split('_')[-2]

            # Read in fixed width weather data
            df = pd.read_fwf(name,
                             names=colnames,
                             colspecs=colspecs,
                             header=None,
                             index_col='time',
                             encoding='latin_1',
                             dtype={'temp': int, 'precip': str},
                             parse_dates=True,
                             date_parser=dateparse)

            # Remove duplicated hours
            df[df.index_duplicated(keep='first') == False]

            


