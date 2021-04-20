"""Make cultivar files."""

import os
import argparse
import pandas as pd

from ideotype.init_params import params_sample


# Setup argparse for terminal control
a = argparse.ArgumentParser(
    description='make cultivar file'
)

a.add_argument(
    'run_name',
    type=str,
    help='run name for batch of simulation experiments')

# fetch args
args = a.parse_args()

N_samples = 300
problem, param_values = params_sample(N_samples)
df_params = pd.DataFrame(param_values.round(2), columns=problem['names'])

fpath_save = os.path.join(os.path.expanduser('~'),
                          'upscale', 'params', f'param_{args.run_name}.csv')

if os.path.exists(fpath_save):
    print('cultivar file exists!')
else:
    df_params.to_csv(fpath_save, index=False)
