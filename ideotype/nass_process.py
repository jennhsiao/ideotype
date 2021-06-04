"""Read & process USDA NASS data."""

import os

import pandas as pd
import numpy as np
import yaml

from ideotype import DATA_PATH


def read_nass():
    """
    Read & process USDA NASS county-level data.

    * note:
    * planting area units: ha
    * yield units: tons/ha

    Returns
    -------
    df_nass : pd.DataFrame

    """
    # Read in file paths
    fpaths_nass = os.path.join(DATA_PATH,
                               'files',
                               'filepaths_nass.yml')
    with open(fpaths_nass) as pfile:
        dict_fpaths = yaml.safe_load(pfile)

    # Assign file paths
    fpath_stateid = os.path.join(DATA_PATH, *dict_fpaths['state_id'])
    fpath_countyid = os.path.join(DATA_PATH, *dict_fpaths['county_id'])
    fpath_lat = os.path.join(DATA_PATH, *dict_fpaths['lat_county'])
    fpath_lon = os.path.join(DATA_PATH, *dict_fpaths['lon_county'])
    fpath_maizeyield = os.path.join(DATA_PATH, *dict_fpaths['maize_yield'])
    fpath_maizearea = os.path.join(DATA_PATH, *dict_fpaths['maize_area'])

    # Read in state & county id
    state_id = pd.read_csv(fpath_stateid, sep='\s+', header=None)  # noqa
    state_id = state_id.transpose()
    state_id.columns = ['state_id']
    county_id = pd.read_csv(fpath_countyid, sep='\s+', header=None)  # noqa
    county_id = county_id.transpose()
    county_id.columns = ['county_id']

    # Read in lat & lon information
    lat_county = pd.read_csv(fpath_lat, sep='\s+', header=None)  # noqa
    lat_county = lat_county.transpose()
    lat_county.columns = ['lat']
    lon_county = pd.read_csv(fpath_lon, sep='\s+', header=None)  # noqa
    lon_county = lon_county.transpose()
    lon_county.columns = ['lon']

    # Read in maize yield
    maize_yield = pd.read_csv(fpath_maizeyield, sep='\s+', header=None)  # noqa
    years = np.arange(1910, 2015)
    maize_yield.columns = years

    # Read in maize area
    maize_area = pd.read_csv(fpath_maizearea, sep='\s+', header=None)  # noqa
    years = np.arange(1910, 2015)
    maize_area.columns = years
    # convert dataframe from wide-form into long-form
    maize_area = maize_area.melt(var_name='year', value_name='area')
    maize_area = maize_area.drop(['year'], axis=1)

    # Concatenate all data and melt dataframe into long-form
    df = pd.concat([state_id, county_id,
                    lat_county, lon_county, maize_yield], axis=1)
    df = pd.melt(df, id_vars=['state_id', 'county_id', 'lat', 'lon'],
                 var_name='year', value_name='yield')

    # Add maize planting area to dataframe
    df_nass = pd.concat([df, maize_area], axis=1)

    return df_nass


def read_irri():
    """
    Read & process NASS cropping area & irrigation data.

    Percent irrigated area calculated across four censue years
    1997, 2002, 2007, and 2012.
    * irrigated area - acres
    * crop area - acres
    * percent irrigated - %

    Returns
    -------
    df_irri : pd.DataFrame

    """
    # Read in file paths
    fpaths_nass = os.path.join(DATA_PATH,
                               'files',
                               'filepaths_nass.yml')
    with open(fpaths_nass) as pfile:
        dict_fpaths = yaml.safe_load(pfile)

    # Assign file paths
    fpath_irri_state = os.path.join(DATA_PATH, *dict_fpaths['irri_state'])
    fpath_irri_county = os.path.join(DATA_PATH, *dict_fpaths['irri_county'])
    fpath_croparea = os.path.join(DATA_PATH, *dict_fpaths['area_planted'])
    fpath_irri_area = os.path.join(DATA_PATH, *dict_fpaths['irri_area'])

    # Read in irrigation dataset
    state_id = pd.read_csv(fpath_irri_state, sep='\s+', header=None).iloc[0, :]  # noqa
    county_id = pd.read_csv(fpath_irri_county, sep='\s+', header=None).iloc[0, :]  # noqa
    crop_area = pd.read_csv(fpath_croparea, sep='\s+', header=None)  # noqa
    irri_area = pd.read_csv(fpath_irri_area, sep='\s+', header=None)  # noqa

    # Average cropping area & irrigated area across 4 census years
    crop_area_mean = crop_area.mean(axis=1)
    irri_area_mean = irri_area.mean(axis=1)
    df_irri = pd.DataFrame({'state_id': state_id,
                            'county_id': county_id,
                            'perct_irri': (
                                irri_area_mean/crop_area_mean)*100})

    return df_irri


def nass_summarize(year_start=1961, year_end=2005):
    """
    Summarize NASS data for specified years.

    Parameters
    ----------
    year_start : int
        Start year for data summary.
    year_end : int
        End year for data summary.

    """
    df_nass = read_nass()
    df_irri = read_irri()

    # Subset to include specified year(s)
    df_nass_sub = df_nass.query(
        f'(year >= {year_start}) & (year <= {year_end})')
    df_nass_sub.reset_index(drop=True, inplace=True)

    # Group df_nass to get max maize planting area & mean yield for maize
    df_nass_maxarea = df_nass_sub.groupby(
        ['state_id', 'county_id']).max()[['lat', 'lon', 'area']]
    df_nass_meanyield = df_nass_sub.groupby(
        ['state_id', 'county_id']).mean()[['yield']]
    df_nass_grouped = pd.concat(
        [df_nass_maxarea, df_nass_meanyield], axis=1).reset_index()

    # Merge maize data with census irrigation data
    df_nass_summary = pd.merge(df_nass_grouped, df_irri,
                               how='left', on=['state_id', 'county_id'])

    return(df_nass_summary)
