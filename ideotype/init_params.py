"""Sample parameter for ensemble runs."""
# TODO: need to make sure parameter file name is generated systematically
# since other modules (e.g. sql_insert insert_paras) parse out
# run_name based on parameter file name

import numpy as np
from SALib.sample import latin

from ideotype.utils import unfold


def params_sample(N_sample):
    """
    Sample parameter through LSH.

    - Phenology:
        - juv_leaves: 15-25 (Padilla and Otegui, 2005)
        - staygreen: 3-8 (still looking for a literature-based value)
        - rmax_ltir: 0.978 (leaves/day) (Kim et al., 2012)
        - rmax_ltar: 0.53 (leaves/day) (Kim et al., 2012)
            - check out (Padilla and Otegui, 2005) for value range
            - Have LTAR vary based on LTIR (e.g. LTAR = 0.5*LTIR)
    - Morphology:
        - LM_min: Length characteristic of longest leaf in canopy (cm)
        - LAF: leaf angle factor for corn leaves (1.37)
    - Physiology:
        - vcm25: 60 µmol/m2s (von Caemmerer, 2000, Table 2)
        - vpm25: 120 µmol/m2s (von Caemmerer, 2000, Table 2), 60 µmol/m2s
          (Kim et al., 2007, Table 4,
          Soo says don't use this, caveats in measurements)
        - go: 0.096 (Yang et al., 2009)
        - g1: 10.055 (Yang et al., 2009);
          Table 1 (Miner et al., 2017): 3.06-3.23 (BB model),
          6.502-9.482 (BBL model)
        - psi_potential:
        - Topt:
    - Management:
        - gdd:
        - row_space:
        - population:

    """
    problem = {
        'num_vars': 10,
        'names': ['juv_leaves',
                  'staygreen',
                  'rmax_ltir',
                  ],
        'bounds': [[],  # juv_leaves
                   [],  # 
                   []]

    }

    param_values = latin.sample(problem, N_sample)
    