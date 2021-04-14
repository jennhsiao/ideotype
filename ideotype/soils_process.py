"""Process SSURGO soil database to create site-specific soil files."""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import compress


def bin_depth(df_soils):
    """
    Bin soil into 5 depth categories.

    Parameters
    ----------
    df_soils : pd.DataFrame

    """
    depths = [[-1, 0],
              [0, 50],
              [50, 100],
              [100, 150],
              [150, 250]]
    depth_categories = [0, 50, 100, 150, 200]
    df_soils['depth_category'] = np.nan

    for depth, depth_category in zip(depths, depth_categories):
        df_soils.loc[
            (df_soils.depth > depth[0]) & (df_soils.depth <= depth[1]),
            'depth_category'] = depth_category

    return df_soils


def merge_texture(df_soils, df_textures):
    """
    Merge texture info (from R) into main soil file.

    Parameters
    ----------
    df_soils : pd.DataFrame
    df_texture : pd.DataFrame

    """
    textures = []
    for i in np.arange(df_textures.shape[0]):
        texture = df_textures.iloc[i]
        try:
            texture = texture[texture == 1].index[0]
            textures.append(texture)
        except IndexError:
            texture = 'ambiguous'
            textures.append(texture)

    df_soils_copy = df_soils.copy()
    df_soils_copy['texture'] = textures

    return df_soils_copy


def texture_profile(df_soils):
    """
    Assign mean texture profile for soil categories.

    - Cl: clay
    - SiCl: silty clay
    - SaCl: sandy clay
    - ClLo: clay loam
    - SiClLo: silty clay loam
    - SaClLo: sandy clay loam
    - Lo: loam
    - SiLo: silty loam
    - SaLo: sandy loam
    - Si: silt
    - LoSa: loamy sand
    - Sa: sand

    Parameters
    ----------
    df_soils : pd.DataFrame

    Returns
    -------
    df_texture : pd.DataFrame

    """
    df_texture = df_soils.groupby(['texture', 'depth_category']).mean()
    df_texture = df_texture[['sand', 'silt', 'clay',
                             'OM', 'dbthirdbar', 'th33', 'th1500']]
    df_texture.OM = df_texture['OM']/100
    df_texture.th33 = df_texture['th33']/100
    df_texture.th1500 = df_texture['th1500']/100

    return df_texture


def texture_prevalence(df_soils, depth1, depth2, sort_column='cokey'):
    """
    Order soil texture based on texture prevalence.

    Parameters
    ----------
    df_soils : pd.DataFrame
    depth1 : int
        First soil depth category to include.
        0.0, 50.0, 100.0, 150.0, 200.0
    depth2 : int
        Second soil depth category to include.
        0.0, 50.0, 100.0, 150.0, 200.0

    Returns
    -------
    df_texture_prevalence : pd.DataFrame

    """
    df_soils_depth = df_soils.query(
        f'(depth_category == {depth1}) | (depth_category == {depth2})')
    df_texture_count = df_soils_depth.groupby('texture').count()
    df_texture_prevalence = pd.DataFrame(df_texture_count.sort_values(
        by=sort_column, axis=0, ascending=False).index)

    return df_texture_prevalence


def assign_texture(df_soils, df_sites, depth1, depth2, n_nearbysites):
    """
    Assign soil texture for each simulation site.

    Parameters
    ----------
    df_soils : pd.DataFrame
    df_sites : pd.DataFrame
    depth1 : int
        First soil depth category to include.
        0.0, 50.0, 100.0, 150.0, 200.0
    depth2 : int
        Second soil depth category to include.
        0.0, 50.0, 100.0, 150.0, 200.0
    n_nearbysites : int

    Returns
    -------
    list_texture : list
        List of textures for all simulation sites.

    """
    sites = df_sites.site
    df_soils_depth = df_soils.query(
        f'(depth_category == {depth1}) | (depth_category == {depth2})')
    list_texture = []

    df_texture_ordered = texture_prevalence(df_soils, depth1, depth2)

    for site in sites:
        lat = float(df_sites[df_sites.site == site].lat)
        lon = float(df_sites[df_sites.site == site].lon)
        # calculate Euclidean distance
        dist = list(enumerate(np.sqrt((lat - df_soils_depth.lat)**2 + (
            lon - (df_soils_depth.lon))**2)))
        df_dist = pd.DataFrame(dist, columns=['rownum', 'distance'])
        # select the nearest n soil sites
        rows = list(df_dist.nsmallest(n_nearbysites, 'distance').rownum)
        # summarize the textures for those sites and order by prevalance
        textures = Counter(df_soils_depth.iloc[rows].texture).most_common()
        # select out only texture counts
        texture_counts = [item[1] for item in textures]

        if len(textures) == 1:
            # when there's only one texture type
            texture = textures[0][0]
            list_texture.append(texture)

        elif len(textures) > 1 and texture_counts[0] != texture_counts[1]:
            # when there's more than one texture type
            # but only one dominant type
            texture = textures[0][0]
            list_texture.append(texture)

        else:
            # when there's more than one texture type
            # and there's a tie between dominant texture types
            maxcount = max(texture_counts)
            # create list with True/False booleans to filter out tied textures
            textures_select = [
                textures[item][1] == maxcount for item in np.arange(
                    len(texture_counts))]
            # filter out tied textures
            textures_selected = list(compress(textures, textures_select))
            textures_selected = [
                textures_selected[item][0] for item in np.arange(
                    len(textures_selected))]
            # identify prevalence between the tied textures
            texture_prevalance = [df_texture_ordered[
                df_texture_ordered.texture == textures_selected[item]].
                index.values[0] for item in np.arange(len(textures_selected))]
            # assing final texture based on df_texture_ordered
            texture = df_texture_ordered[
                df_texture_ordered.index == min(
                    texture_prevalance)].texture.values[0]
            list_texture.append(texture)

    return list_texture
