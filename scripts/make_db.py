"""
Make database and insert values into tables.

Tables include:
- site_inso
- params
- weadata
- sims

"""
import os
import argparse

from ideotype.sql_declarative import create_table
from ideotype.sql_insert import (insert_siteinfo, insert_params,
                                 insert_weadata, insert_sims)
from ideotype.wflow_setup import read_inityaml
from ideotype import DATA_PATH

a = argparse.ArgumentParser(
    description='additional info to insert tables into DB'
)

a.add_argument(
    'run_name',
    type=str,
    help='run name for batch of simulation experiments'
)

args = a.parse_args()

# path for database
# fpath_db = '/home/disk/eos8/ach315/upscale/db/ideotype.db'
fpath_db = os.path.join(os.path.expanduser('~'),
                        'upscale',
                        'db',
                        f'{args.run_name}.db')

# make DB if DB doesn't exist yet
if os.path.exists(fpath_db):
    print('DB exists!')
else:
    create_table(fpath_db)

# read in init yaml file info for specific run
init_dict = read_inityaml(args.run_name)

# setup pointers to db table files
fpath_siteinfo = os.path.join(DATA_PATH,
                              'sites',
                              init_dict['site_summary'])
fpath_params = os.path.join(DATA_PATH,
                            'params',
                            init_dict['path_params'])
dirct_weadata = os.path.join(os.path.expanduser('~'),
                             'upscale',
                             *init_dict['path_wea'])
dirct_sims = os.path.join(os.path.expanduser('~'),
                          'upscale',
                          'sims',
                          f'{args.run_name}')

# insert values into table for specific run
insert_siteinfo(fpath_siteinfo, fpath_db)
insert_params(fpath_params, fpath_db, args.run_name)
insert_weadata(dirct_weadata, fpath_db, n_savefiles=500)
insert_sims(dirct_sims, fpath_db, args.run_name, n_savefiles=1000)
