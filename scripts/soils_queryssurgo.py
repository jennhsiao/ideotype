"""Query SSURGO soil data."""

import pandas as pd

from ideotype.soils_query import soilquery
from ideotype.nass_process import nass_summarize

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
