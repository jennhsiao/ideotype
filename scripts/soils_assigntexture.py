"""Process and assing soil texture for each simulation site."""

import os
import yaml
import pandas as pd
from ideotype import DATA_PATH
from ideotype.soils_process import (bin_depth,
                                    merge_texture,
                                    texture_profile,
                                    assign_texture)

# Read in filepaths
file_paths = os.path.join(DATA_PATH, 'files', 'filepaths_soils.yml')
with open(file_paths, 'r') as pfile:
    dict_files = yaml.safe_load(pfile)

# Set up relevant files
file_sites = os.path.join(DATA_PATH, 'files', dict_files['sites'])
file_soils = os.path.join(DATA_PATH, 'files', dict_files['soils'])
file_soiltextures = os.path.join(DATA_PATH, 'files', dict_files['soiltexture'])

# Read in files into dataframes
df_sites = pd.read_csv(file_sites, dtype={'site': str})
df_soils = pd.read_csv(file_soils, dtype={'sgroup': str})
df_soiltextures = pd.read_csv(file_soiltextures)

# Bin soil into 5 soil depth categories
df_soils = bin_depth(df_soils)

# Include soil texture info into df_soils
df_soils = merge_texture(df_soils, df_soiltextures)

# Create texture profile for all soil texture groups
df_textureprofile = texture_profile(df_soils)

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

# Save out relevant files
#df_textureprofile.to_csv('~/upscale/weadata/soils_textureprofile.csv')  # noqa
#df_sites_copy.to_csv('~/upscale/weadata/site_summary.csv', index=False)  # noqa
