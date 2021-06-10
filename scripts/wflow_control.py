"""
Generate init text files for CONTROL maizsim simulations.

Two control sims
- control_fixpd: fixed planting date
- control_dympd: dynamice planting date

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

# Fetch args
args = a.parse_args()

# Workflow
log_fetchinfo(args.run_name)
make_dircts(args.run_name, cont_cvars=False)
make_inits(args.run_name, cont_cvars=False)
make_cultivars(args.run_name, cont_cvars=False)
make_runs(args.run_name, cont_cvars=False)
make_jobs(args.run_name, cont_cvars=False)
make_subjobs(args.run_name)
