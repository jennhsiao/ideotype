"""Read and combine raw weather data for future climate."""

import argparse
import os

import pandas as pd

from ideotype.weafile_process import (read_wea,
                                      wea_combine,
                                      wea_preprocess,
                                      wea_summarize,
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

# Step 1: Read weather files
print('*** step 1: read wea data')
read_wea(1961, 2010, args.future_year)
# no need to read_solrad since we assume no change in solrad
# and that data is directly copied from /weadata/process

# Step 2: Combine weather data for all years
print('*** step 2: combine wea data')
basepath = (f'/home/disk/eos8/ach315/upscale/weadata/'
            f'process_f{args.future_year}')
wea_combine(basepath)

# Step 3: Preprocess combined weather files
print('*** step 3: preprocess combined wea data')
# paths
basepath_siteyears = os.path.join(DATA_PATH, 'siteyears')
wea_folder = f'f{args.future_year}'
outpath = os.path.join(
    '/home/disk/eos8/ach315/upscale/weadata/data', wea_folder)
# preprocess
df_temp, df_rh, df_precip, df_solrad = wea_preprocess(basepath)

# Step 4. Summarize site-year weather info
print('*** step 4: summarize site-year wea info')
# parameters & paths
gseason_start_weasummary = 3
gseason_end_weasummary = 10
outpath_wea = os.path.join(DATA_PATH, 'wea')
# the step of determining site-years is already done when making
# weather files for 'present' sims, so did not repeat for future climate
# instead, just use the already existing site-years
siteyears_filtered = pd.read_csv(
    os.path.join(basepath_siteyears, 'siteyears_control_fixpd.csv'), dtype=str)
# summarize wea
df_wea = wea_summarize(siteyears_filtered,
                       df_temp, df_rh, df_precip, df_solrad,
                       gseason_start_weasummary, gseason_end_weasummary)
df_wea.to_csv(
    os.path.join(outpath_wea, f'wea_summary_f{args.future_year}.csv'),
    index=False)

# Step 5: Convert met data into individual maizsim weather data files
print('*** step 5: make wea file')
make_weafile(siteyears_filtered, df_temp, df_rh, df_precip, df_solrad, outpath,
             climate_treatment=args.future_year)
