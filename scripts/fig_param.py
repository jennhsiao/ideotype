"""Fig. Param."""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from palettable.colorbrewer.sequential import YlGnBu_8

df_params = pd.read_csv(
    '/home/disk/eos8/ach315/upscale/params/param_fixpd.csv')
outpath = '/home/disk/eos8/ach315/upscale/figs/'

x = df_params.values
minmax_scale = preprocessing.MinMaxScaler()
x_scaled = minmax_scale.fit_transform(x)
df_scaled = pd.DataFrame(x_scaled).transpose()
df_scaled.index = ['g1',
                   'Vcmax',
                   'Jmax',
                   'phyf',
                   'SG',
                   'gleaf',
                   'LTAR',
                   'LM',
                   'LAF',
                   'gdd',
                   'pop']

# All params
fig, ax = plt.subplots(figsize=(30, 5))
ax = sns.heatmap(df_scaled, cmap=YlGnBu_8.mpl_colormap)
plt.xticks(fontweight='light', fontsize=12)
plt.yticks(rotation=0, fontweight='light', fontsize=12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
fig.subplots_adjust(left=0.2, bottom=0.45)

plt.savefig(os.path.join(outpath, 'params_all.png'),
            format='png', dpi=800)

# Small params fig
fig, ax = plt.subplots(figsize=(5, 5))
df_sub = df_scaled.iloc[:, :15]
ax = sns.heatmap(df_sub, cmap=YlGnBu_8.mpl_colormap)
plt.xticks(fontweight='light', fontsize=12)
plt.yticks(rotation=0, fontweight='light', fontsize=12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
fig.subplots_adjust(left=0.2, bottom=0.45)

plt.savefig(os.path.join(outpath, 'params_small.png'),
            format='png', dpi=800)
