"""Sample parameter for ensemble runs."""
# TODO: need to make sure parameter file name is generated systematically
# since other modules (e.g. sql_insert insert_paras) parse out
# run_name based on parameter file name

from SALib.sample import latin
from ideotype.wflow_setup import read_inityaml


def params_sample(run_name, N_sample, yamlfile=None):
    """
    Sample parameter through LSH.

    - Physiology:
        - g1: Ball-Berry gs model slope
        - Vcmax: Max RUBISCO capacity
        - Jmax: Max electron transport rate
        - phyf: Leaf water potential triggering stomatal closure (MPa)
    - Phenology:
        - SG: Duration that leaves maintain active function after maturity
        - gleaf: Generic total leaf number
        - LTAR: Max leaf tip appearance rate (leaves/day)
    - Morphology:
        - LM: Length characteristic of longest leaf in canopy (cm)
        - LAF: leaf angle factor for corn leaves (1.37)
    - Management:
        - gdd: growing degree days accumulated before sowing
        - pop: plants/m2

    Parameters
    ----------
    N_sample : number of samples to generate.

    """
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    names = list(dict_setup['params'].keys())
    bounds = list(dict_setup['params'].values())

    problem = {
        'num_vars': len(names),
        'names': names,
        'bounds': bounds
    }

    param_values = latin.sample(problem, N_sample)

    return problem, param_values
