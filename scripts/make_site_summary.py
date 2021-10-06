"""Make site_summary.csv file."""
# TODO: 210630 - to be continued, but is probably not a current priority

import os

import pandas as pd

from ideotype import DATA_PATH


df_stations_info = pd.read_csv(os.path.join(DATA_PATH,
                                            'sites',
                                            'stations_info_9110.csv'))

# Continental US site boundaries
lat_min = 19
lat_max = 53
lon_min = -123
lon_max = -72

df_stations = df_stations_info[(df_stations_info['ISH_LON(dd)'] > -123) &
                               (df_stations_info['ISH_LON(dd)'] < -72) &
                               (df_stations_info['ISH_LAT (dd)'] > 19) &
                               (df_stations_info['ISH_LAT (dd)'] < 53)]
