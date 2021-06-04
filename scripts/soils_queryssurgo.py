"""Query SSURGO soil data."""

import os
import yaml

import pandas as pd

from ideotype import DATA_PATH
from ideotype.soils_query import soilquery
from ideotype.nass_process import nass_summarize

# Read in filepaths
#file_paths = os.path.join(DATA_PATH, 'files', 'filepaths_soils.yml')
#with open(file_paths, 'r') as pfile:
#    dict_files = yaml.safe_load(pfile)

# Read in NASS site info
#file_nass = os.path.join(DATA_PATH, 'files', dict_files['nass_siteinfo'])
#df_nass = pd.read_csv(file_nass, index_col=0)

df_nass = nass_summarize()
lats = df_nass.lat
lons = df_nass.lon

# Set up soil query dataframe
df_soils = pd.DataFrame()

# Query soil info for NASS sites
for lat, lon in zip(lats, lons):
    try:
        df_soil = soilquery(round(lat, 2), round(lon, 2))
        df_soil['lat'] = lat
        df_soil['lon'] = lon
        df_soils = df_soils.append(df_soil)
    except:  # noqa
        print(f'{lat}/{lon}')

# Drop rows that include nan
df_soils.dropna(axis=0, inplace=True)

# Include only sites where sand/silt/clay adds up to 100%
df_soils = df_soils[df_soils.sand + df_soils.silt + df_soils.clay == 100]
df_soils.reset_index(drop=True, inplace=True)

#df_soils.to_csv('~/upscale/weadata/soils_nass.csv', index=False)  # noqa
