"""Process SSURGO soil database to create site-specific soil files."""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import compress


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
    df_texture = df_soils.groupby('texture').mean()
    df_texture = df_texture[['sand', 'silt', 'clay',
                             'OM', 'dbthirdbar', 'th33']]

    return df_texture


def texture_prevalence(df_soils, depth, sort_column='cokey'):
    """
    Order soil texture based on texture prevalence.

    Parameters
    ----------
    df_soils : pd.DataFrame
    depth : int
        Soil depth category.
        0.0, 50.0, 100.0, 150.0, 200.0

    Returns
    -------
    df_texture_ordered : pd.DataFrame

    """
    df_soils_depth = df_soils.query(f'depth_category == "{depth}"').dropna()
    df_texture_count = df_soils_depth.groupby('texture').count()
    df_texture_ordered = pd.DataFrame(df_texture_count.sort_values(
        by=sort_column, axis=0, ascending=False).index)

    return df_texture_ordered


def assign_texture(df_soils, df_sites, depth, n_nearbysites):
    """
    Assign soil texture for each simulation site.

    Parameters
    ----------
    df_soils : pd.DataFrame
    df_sites : pd.DataFrame
    depth : int
        Soil depth category.
        0.0, 50.0, 100.0, 150.0, 200.0
    n_nearbysites : int

    Returns
    -------
    list_texture : list
        List of textures for all simulation sites.

    """
    sites = df_sites.site
    df_soils_depth = df_soils.query(f'depth_category == "{depth}"').dropna()
    list_texture = []

    df_texture_ordered = texture_prevalence(df_soils, depth)

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
