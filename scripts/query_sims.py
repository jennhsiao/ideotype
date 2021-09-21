"""Query sims from database."""

import os
import argparse

import numpy as np

from ideotype.sql_query import (query_yield,
                                query_phys,
                                query_carbon,
                                query_pheno,
                                query_leaves,
                                query_mass,
                                query_waterstatus,
                                query_waterstatus_sum,
                                query_waterpotential)

# Set up argparse for terminal control
a = argparse.ArgumentParser(
    description='additional info to query sims'
)

a.add_argument(
    'run_name',
    type=str,
    help='run name for batch of simulation experiments')

args = a.parse_args()

# Set up databse path
fpath_db = f'/home/disk/eos8/ach315/upscale/db/{args.run_name}.db'
phenos = np.arange(100).tolist()

# Query yield
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_yield.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query yield')
    query, results, df = query_yield(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query physiology
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_phys.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query phys')
    query, results, df = query_phys(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query carbon gained
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_carbon.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query carbon')
    query, results, df = query_carbon(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query mass
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_mass.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query mass')
    query, results, df = query_mass(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query phenology
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_pheno.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query pheno')
    query, results, df = query_pheno(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query morphology
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_leaves.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query leaves')
    query, results, df = query_leaves(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query water deficit
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_waterdeficit.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query waterstatus')
    query, results, df = query_waterstatus(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query water deficit sum
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_waterdeficit_sum.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query water status sum')
    query, results, df = query_waterstatus_sum(fpath_db, phenos)
    df.to_csv(csvfile, index=False)

# Query water potential
csvfile = (f'/home/disk/eos8/ach315/ideotype/ideotype/data/sims/'
           f'sims_{args.run_name}_waterpotential.csv')
if os.path.isfile(csvfile):
    print(f'{csvfile} exists already!')
else:
    print('query water potential')
    time = 5
    query, results, df = query_waterpotential(fpath_db, phenos, time)
    df.to_csv(csvfile, index=False)
