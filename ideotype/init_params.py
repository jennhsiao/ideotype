"""Sample parameter for ensemble runs."""
# TODO: need to make sure parameter file name is generated systematically
# since other modules (e.g. sql_insert insert_paras) parse out
# run_name based on parameter file name

from SALib.sample import latin


def params_sample(N_sample):
    """
    Sample parameter through LSH.

    - Physiology:
        - g1: Ball-Berry slope
            * 3.06-3.23 (Miner et al., 2017, Table 1 - BB model)
        - vcmax:
        - jmax:
        - phyf:
    - Phenology:
        - staygreen:
        - juv_leaves:
        - rmax_ltar:
            * note that rmax_ltir should be 2*rmax_ltar
    - Morphology:
        - lm_min: Length characteristic of longest leaf in canopy (cm)
        - laf: leaf angle factor for corn leaves (1.37)
    - Management:
        - gdd: growing degree days accumulated before sowing
        - pop: plants/m2

    Parameters
    ----------
    N_sample : number of samples to generate.

    """
    problem = {
        'num_vars': 11,  # TODO: still need to finalize
        'names': ['g1',
                  'vcmax',
                  'jmax',
                  'phyf'
                  'staygreen',
                  'juv_leaves',
                  'rmax_ltar',
                  'lm_min',
                  'laf',
                  'gdd',
                  'pop'
                  ],
        'bounds': [[2, 6],  # g1
                   [65, 80]  # Vcmax
                   [350, 420]  # Jmax
                   [-3, -1]  # phyf
                   [2, 6],  # staygreen
                   [11, 25],  # juv_leaves
                   [0.4, 0.8],  # rmax_ltar
                   [80, 120],  # LM_min
                   [0.9, 1.4],  # LAF
                   [80, 160],  # gdd
                   [6, 14],  # population
                   ]
    }

    param_values = latin.sample(problem, N_sample)

    return problem, param_values
