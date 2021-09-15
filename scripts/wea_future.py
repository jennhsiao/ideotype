"""Read and combine raw weather data for future climate."""

import argparse
import os

import pandas as pd

from ideotype.weafile_process import (read_wea,
                                      wea_combine,
                                      wea_preprocess,
                                      make_weafile)
from ideotype import DATA_PATH


# Step 0: set up args and parameters
# Setup argparse for terminal control
a = argparse.ArgumentParser(
    description='additional info to setup project directories and files'
)

a.add_argument(
    'future_year',
    type=str,
    help='future time point for climate projections - 2050 or 2100')

# Fetch args
args = a.parse_args()

# Run name
run_name = 'control_fixpd'
wea_folder = str(args.future_year)

# Step 1: process weather files
# Read in raw weather data file
read_wea(1961, 2010, args.future_year)
# no need to read_solrad since we assume no change in solrad
# and that data is directly copied from /weadata/process

# Step 2: Combine weather data for all years
basepath = f'/home/disk/eos8/ach315/upscale/weadata/process_{args.future_year}'
wea_combine(basepath)

# Step 3: write out weather files
# Set up paths
basepath_siteyears = os.path.join(DATA_PATH, 'siteyears')
outpath = os.path.join(
    '/home/disk/eos8/ach315/upscale/weadata/data', wea_folder)

# Preprocess combined weather data
df_temp, df_rh, df_precip, df_solrad = wea_preprocess(basepath)

# Read in siteyears_filtered
# the step of determining site-years is already done when making
# weather files for 'present' sims, so did not repeat for future climate
# instead, just use the already existing site-years
siteyears_filtered = pd.read_csv(
    os.path.join(basepath_siteyears, f'siteyears_{run_name}.csv'), dtype=str)

# Convert met data into individual maizsim weather data files
make_weafile(
    siteyears_filtered, df_temp, df_rh, df_precip, df_solrad, outpath)
