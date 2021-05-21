"""Read, prcess, and output maizsim weather data."""
from ideotype.weafile_process import (read_wea,
                                      read_solrad,
                                      wea_combine,
                                      wea_preprocess,
                                      wea_siteyears,
                                      wea_filter)

# Read in raw weather data file
read_wea(1961, 2010)
read_solrad(1961, 2010)

# Combine weather data for all years
basepath = '/home/disk/eos8/ach315/upscale/weadata/process'
wea_combine(basepath)

# Preprocess combined weather data
df_temp, df_rh, df_precip, df_solrad = wea_preprocess(basepath)

# Select valid site-years
crthr = 2
siteyears = wea_siteyears(df_temp, df_precip, df_solrad, crthr)

# Filter site-years based on area, irrigation, & estimated pdate
area = 10000
irri = 50
siteyears_filtered = wea_filter(siteyears, area, irri)

# Convert met data into individual maizsim weather data files
