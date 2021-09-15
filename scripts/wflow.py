"""
Generate init text files for maizsim simulations.

Maizsim has a total of 16 intput files.
The files listed here are only ones that require:
1. Customization for individual site-years, or
2. Multiple sets for parametetr sensitivity testing.

"""
import argparse
from ideotype.log import log_fetchinfo
from ideotype.wflow_setup import (make_dircts, make_inits, make_cultivars,
                                  make_runs, make_jobs, make_subjobs)

# Setup argparse for terminal control
a = argparse.ArgumentParser(
    description='additional info to setup project directories and files'
)

a.add_argument(
    'run_name',
    type=str,
    help='run name for batch of simulation experiments')

a.add_argument(
    '--dryrun',
    type=bool,
    default=False,
    help=('print info on directories that are going to be made'
          'rather than actually making them')
)

# fetch args
args = a.parse_args()

# SETP 0.0: document log file
print('*** step 0.0: document log file')
log_fetchinfo(args.run_name)

# STEP 0.1: setup directories
print('*** step 0.1: set up directories')
make_dircts(args.run_name)

# STEP 1: create maizsim initial files
# init.txt, time.txt, climate.txt, management.txt
print('*** step 1: make inits')
make_inits(args.run_name)

# STEP 2: create cultivar files
print('*** step 2: make cultivar files')
make_cultivars(args.run_name)

# STEP 3: create run.txt files
print('*** step 3: make run files')
make_runs(args.run_name)

# step 4: create job files that execute batch of run files
print('*** step 4: make job files')
make_jobs(args.run_name)

# step 5: create bash script that automates qsub jobs
print('*** step 5: make subjob file')
make_subjobs(args.run_name)
