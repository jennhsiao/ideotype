"""Determine soil texture class for each simulation site."""

import os
import yaml
import pandas as pd
from ideotype import DATA_PATH
from ideotype.soils_process import assign_texture

# Read in filepaths
file_paths = os.path.join(DATA_PATH, 'files', 'filepaths_fixpd.yml')
with open(file_paths, 'r') as pfile:
    dict_files = yaml.safe_load(pfile)

# Set up relevant files
file_soil = os.path.join(DATA_PATH, 'files', dict_files['soils'])
file_sites = os.path.join(DATA_PATH, 'files', dict_files['sites'])

# Read in files into dataframes
df_soils = pd.read_csv(file_soil, index_col=0, dtype={'sgroup': str})
df_sites = pd.read_csv(file_sites, dtype={'site': str})

# Exclude sand & clay from df_soils
df_soils = df_soils.query('(texture != "Cl") & (texture != "Sa")')

# Assign soil texture groups for each simulation site
# Soil texture for each site determined based on top two soil layers
depth1 = 0
depth2 = 50
n_nearbysites = 25
texture_sites = assign_texture(
    df_soils, df_sites, depth1, depth2, n_nearbysites)

# Append soil texture info to site info
df_sites_copy = df_sites.copy()
df_sites_copy['texture'] = texture_sites

#df_sites_copy.to_csv('~/upscale/weadata/site_summary.csv', index=False)
