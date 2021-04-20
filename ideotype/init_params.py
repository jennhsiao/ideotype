"""Sample parameter for ensemble runs."""
# TODO: need to make sure parameter file name is generated systematically
# since other modules (e.g. sql_insert insert_paras) parse out
# run_name based on parameter file name

from SALib.sample import latin


def params_sample(N_sample):
    """
    Sample parameter through LSH.

    - Phenology:
        - juv_leaves: 15-25 (Padilla and Otegui, 2005)
        - staygreen: 3-8 (still looking for a literature-based value)
        - rmax_ltar: 0.53 (leaves/day) (Kim et al., 2012)
            * note that rmax_ltir should be 2*rmax_ltar
            * check out (Padilla and Otegui, 2005) for value range
    - Morphology:
        - LM_min: Length characteristic of longest leaf in canopy (cm)
        - LAF: leaf angle factor for corn leaves (1.37)
    - Physiology:
        - g1: Ball-Berry slope
            * 10.055 (Yang et al., 2009)
            * 3.06-3.23 (Miner et al., 2017, Table 1 - BB model)
            * 6.502-9.482 (Miner et al., 2017, Table 1 - BBL model)
        - psi_potential:
        - Topt:
    - Management:
        - gdd: growing degree days accumulated before sowing
        - row_space: distance between planting rows (m)
        - population: plants/m2

    Parameters
    ----------
    N_sample : number of samples to generate.

    """
    problem = {
        'num_vars': 12,  # TODO: still need to finalize
        'names': ['juv_leaves',
                  'staygreen',
                  'rmax_ltir',
                  'LM_min',
                  'LAF',
                  'g1',
                  'psi_potential',
                  'Q10',
                  'Topt',
                  'gdd',
                  'row_space',
                  'population'
                  ],
        'bounds': [[15, 25],  # juv_leaves
                   [3, 8],  # staygreen
                   [1, 3],  # rmax_ltar
                   [80, 120],  # LM_min
                   [1, 5],  # LAF  # TODO: verify
                   [1, 10],  # g1
                   [-4, -1],  # psi_potential  # TODO: verify
                   [1.2, 3],  # Q10  # TODO: verify
                   [30, 35],  # Topt  # TODO: verify
                   [100, 200],  # gdd  # TODO: verify
                   [0.5, 1.5],  # row_space  # TODO: verify
                   [6, 12],  # population  # TODO: verify
                   ]
    }

    param_values = latin.sample(problem, N_sample)

    return problem, param_values
