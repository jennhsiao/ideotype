"""Write out MAIZSIM weather files."""

import os

import pandas as pd

from ideotype.weafile_process import (wea_preprocess,
                                      make_weafile)
from ideotype import DATA_PATH

# Run name
run_name = 'control_fixpd'
wea_folder = 'control'

# Paths
basepath_wea = '/home/disk/eos8/ach315/upscale/weadata/process'
basepath_siteyears = os.path.join(DATA_PATH, 'siteyears')
outpath = os.path.join(
    '/home/disk/eos8/ach315/upscale/weadata/data', wea_folder)

# Preprocess combined weather data
df_temp, df_rh, df_precip, df_solrad = wea_preprocess(basepath_wea)

# Read in siteyears_filtered
siteyears_filtered = pd.read_csv(
    os.path.join(basepath_siteyears, f'siteyears_{run_name}.csv'), dtype=str)

# Convert met data into individual maizsim weather data files
make_weafile(
    siteyears_filtered, df_temp, df_rh, df_precip, df_solrad, outpath)
