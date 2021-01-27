"""
Generate init text files for maizsim simulations.

Maizsim has a total of 16 intput files.
The files listed here are only ones that require:
1. Customization for individual site-years, or
2. Multiple sets for parametetr sensitivity testing.

"""
import argparse
from ideotype.log import log_fetchinfo
from ideotype.wflow_setup import (make_dircts, make_runs,
                                  make_jobs, make_subjobs)

# TODO: really need to test this code
# TODO: to properly do so, will need to setup test directory to test on
# TODO: add code to mkdir for custom init files here as well
# TODO: update some names to make code more readable
# TODO: test that output directories and run/job files are correct
# TODO: if directories already exist, print to terminal to show that
# TODO: update some hard-coded directories to point to yml file


# Setup argparse for terminal control
a = argparse.ArgumentParser(
    description='help text here ___ '
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
log_fetchinfo(args.run_name)

# STEP 0.1: setup directories
make_dircts(args.run_name)

# STEP 1: create run.txt files
make_runs(args.run_name)

# step 2: create job files that execute batch of run files
make_jobs(args.run_name)

# step 3: create bash script that automates qsub jobs
make_subjobs(args.run_name)
