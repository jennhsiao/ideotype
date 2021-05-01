"""Misc utility functions."""
import os
from datetime import datetime

import pandas as pd
import numpy as np
import pytz
from sklearn.linear_model import LinearRegression


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
    # Es_Tref = 6.11 # saturation vapor pressure at reference temperature (mb)
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


def calc_gdd(temps, temp_base=8, gdd_threshold=100):
    """
    Maize growing season GDD calculation.

    - calculates GDH with base temperature = 8˚C
    - calculated values divided by 24 to correspond to daily values
    - function returns count of point in which gdd exceeds 100
      which can then be used to identify date in which GDD=100 is reached
    - citation: the solar corridor crop system

    Parameters
    ----------
    temps : list
        List of hourly temp for gdd accumulation.
    temp_base : int
        Base temperature for gdd accumulation.
    gdd_threshold : int
        GDD threshold for planting.

    Returns
    -------
    count : int
        The number count in which gdd is reached.

    """
    gdd = 0
    for count, temp in enumerate(temps):
        if gdd > gdd_threshold:
            break
        else:
            if temp-temp_base < 0:
                gdd += 0
            else:
                gdd += (temp-temp_base)/24
    return(count)


def estimate_pdate(basepath, site, year, gdd_threshold=100):
    """
    Estimate planting date for specified location.

    Parameters
    ----------
    basepath : str
        Base path to fetch weather data.
    site : str
        site_id for specified site-year.
    year : int
        Year for specified site-year.
    gdd_threshold : int
        GDD threshold for planting date estimation.

    Returns
    -------
    date_start : str
        Simulation start date in maizsim format.
    date_plant : str
        Simulation plant date in for maizsim format.

    """
    fpath_wea = os.path.join(basepath, f'{site}_{year}.txt')
    df_wea = pd.read_csv(fpath_wea)
    temps = list(df_wea.temp)

    loc = calc_gdd(temps, gdd_threshold)
    jday_plant = df_wea.loc[loc, 'jday']
    jday_start = jday_plant - 14  # start date 2 weeks prior to planting

    # cap earliest start date as Feb 1st
    if jday_plant < 32:
        jday_plant = 32

    date_plant = datetime.strptime(
        f'{year}-{jday_plant}', '%Y-%j').strftime("'%m/%d/%Y'")
    date_start = datetime.strptime(
        f'{year}-{jday_start}', '%Y-%j').strftime("'%m/%d/%Y'")

    return date_start, date_plant


def stomata_waterstress():
    """
    Estimate sf from phyf via linear function.

    Data from Tuzet, Perrier, and Leuning, 2003, Plant Cell Environ.
    Fig. 6a - Difference in stomata water stress response curve.
    sf: sensitivity parameter
    phyf: reference potential

    Returns
    -------
    mod_intercept
    mod_coef

    """
    x = [[-1.2], [-1.9], [-2.6]]
    y = [[4.9], [3.2], [2.3]]
    mod = LinearRegression()
    mod.fit(x, y)
    mod_intercept = mod.intercept_[0]
    mod_coef = mod.coef_[0][0]

    return mod_intercept, mod_coef
