"""Read and combine raw weather data."""

from ideotype.weafile_process import (read_wea,
                                      read_solrad,
                                      wea_combine)

# Read in raw weather data file
read_wea(1961, 2010)
read_solrad(1961, 2010)

# Combine weather data for all years
basepath = '/home/disk/eos8/ach315/upscale/weadata/process'
wea_combine(basepath)
