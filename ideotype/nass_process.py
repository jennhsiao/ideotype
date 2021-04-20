"""Read & process USDA NASS data."""

import os

import pandas as pd
import numpy as np
import yaml

from ideotype import DATA_PATH


def read_nass():
    """
    Read & process USDA NASS county-level data.

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


def summarize_nass():
    """
    Read & process NASS planting area & irrigation data.

    Cropping & irrigated area across four censue years:
    1997, 2002, 2007, and 2012.

    Returns
    -------
    df_nass_summary : pd.DataFrame

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
    df_census = pd.DataFrame({'state_id': state_id,
                              'county_id': county_id,
                              'prect_irri': (
                                  irri_area_mean/crop_area_mean)*100})

    # Read in nass maize data
    df_nass = read_nass()
    # subset to include only years 1961-2005
    df_nass_sub = df_nass.query('year>=1961 & year<=2005')
    df_nass_sub.reset_index(drop=True, inplace=True)
#    years_int = df_nass_sub.year.astype(int)
#    df_nass_sub.year = years_int

    # Group df_nass to get max maize planting area & mean yield for maize
    df_nass_maxarea = df_nass_sub.groupby(
        ['state_id', 'county_id']).max()[['lat', 'lon', 'area']]
    df_nass_meanyield = df_nass_sub.groupby(
        ['state_id', 'county_id']).mean()[['yield']]
    df_nass_grouped = pd.concat(
        [df_nass_maxarea, df_nass_meanyield], axis=1).reset_index()

    # Merge maize data with census irrigation data
    df_nass_census = pd.merge(df_nass_grouped, df_census,
                              how='left', on=['state_id', 'county_id'])

    return df_nass_census
