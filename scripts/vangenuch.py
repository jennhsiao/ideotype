import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def vangenuch(thr, ths, alpha, n):
    """
    Van Genuch equation to estimate water retention curve.

    returns
    -------
    hs : list
        matric potential (kPa)
    ys : list
        soil water content (cm3/cm3)

    """
    m = -(1-1/n)
    hs = np.arange(1, 15000)  # matric potential (kPa)
    y0s = [((1 + alpha*h)**n)**m for h in hs]
    ys = [(ths-thr)*y0 + thr for y0 in y0s]  # soil water content (cm3/cm3)

    return(hs, ys)


# setup
hs_sa, ys_sa = vangenuch(0.027, 0.398, 0.05610, 1.44361)
hs_cl, ys_cl = vangenuch(0.099, 0.516, 0.02436, 1.2045)
hs_cllo, ys_cllo = vangenuch(0.077, 0.45, 0.02147, 1.28374)
hs_lo, ys_lo = vangenuch(0.047, 0.412, 0.01567, 1.35713)
hs_losa, ys_losa = vangenuch(0.027, 0.387, 0.04095, 1.36571)
hs_sacllo, ys_sacllo = vangenuch(0.083, 0.421, 0.04506, 1.35181)
hs_salo, ys_salo = vangenuch(0.038, 0.401, 0.03878, 1.35869)
hs_sicl, ys_sicl = vangenuch(0.147, 0.474, 0.05272, 1.26669)
hs_sicllo, ys_sicllo = vangenuch(0.084, 0.464, 0.01932, 1.2821)
hs_silo, ys_silo = vangenuch(0.047, 0.411, 0.01252, 1.38227)

# visualize
fig = plt.figure(figsize=(7, 5))
ax = plt.gca()

# reference lines for sand & clay
ax.plot(hs_sa, ys_sa, c='grey', label='Sa', linestyle=':')
ax.plot(hs_cl, ys_cl, c='grey', label='Cl', linestyle='--', linewidth=1.2)

# other soil types
ax.plot(hs_cllo, ys_cllo, c='#1b9e77', label='ClLo', linewidth=2)
ax.plot(hs_lo, ys_lo, c='#d95f02', label='Lo', linewidth=2)
ax.plot(hs_losa, ys_losa, c='#7570b3', label='LoSa', linewidth=2)
ax.plot(hs_sacllo, ys_sacllo, c='#e7298a', label='SaClLo', linewidth=2)
ax.plot(hs_salo, ys_salo, c='#66a61e', label='SaLo', linewidth=2)
ax.plot(hs_sicl, ys_sicl, c='#e6ab02', label='SiCl', linewidth=2)
ax.plot(hs_sicllo, ys_sicllo, c='#a6761d', label='SiClLo', linewidth=2)
ax.plot(hs_silo, ys_silo, c='#666666', label='SiLo', linewidth=2)

ax.set_xscale('log')
ax.set_xlabel('matric potential', size=14, fontweight='light')
ax.set_ylabel('soil water content', size=14, fontweight='light')
ax.set_ylim(0.1, 0.5)
ax.legend()

# slope calculation
num = 50  # range of matric potential to calculate slope

dict_m = {
    'SiCl': [round(
        ys_sicl[num]-ys_sicl[0]/hs_sicl[num]-hs_sicl[0], 2)],
    'SiClLo': [round(
        ys_sicllo[num]-ys_sicllo[0]/hs_sicllo[num]-hs_sicllo[0], 2)],
    'ClLo': [round(
        ys_cllo[num]-ys_cllo[0]/hs_cllo[num]-hs_cllo[0], 2)],
    'SaLo': [round(
        ys_salo[num]-ys_salo[0]/hs_salo[num]-hs_salo[0], 2)],
    'SaClLo': [round(
        ys_sacllo[num]-ys_sacllo[0]/hs_sacllo[num]-hs_sacllo[0], 2)],
    'LoSa': [round(
        ys_losa[num]-ys_losa[0]/hs_losa[num]-hs_losa[0], 2)],
    'SiLo': [round(
        ys_silo[num]-ys_silo[0]/hs_silo[num]-hs_silo[0], 2)],
    'Lo': [round(
        ys_lo[num]-ys_lo[0]/hs_lo[num]-hs_lo[0], 2)]}

df_m = pd.DataFrame.from_dict(dict_m, orient='index', columns=['slope'])
df_m['ks'] = [17.56, 23.656, 101.856, 56.964, 68.932, 15.573, 19.993, 34.213]
