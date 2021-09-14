"""Read and combine raw weather data for future climate."""

import argparse

from ideotype.weafile_process import (read_wea,
                                      wea_combine)

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

# Read in raw weather data file
read_wea(1961, 2010, args.future_year)
# no need to read_solrad since we assume no change in solrad
# and that data is directly copied from /weadata/process

# Combine weather data for all years
basepath = '/home/disk/eos8/ach315/upscale/weadata/process_2050'
wea_combine(basepath)
