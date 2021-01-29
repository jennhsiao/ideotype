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
fpath_db = '/home/disk/eos8/ach315/upscale/db/smalldb.db'

# make DB if DB doesn't exist yet
if os.path.exists(fpath_db):
    print('DB already exists!')
else:
    create_table(fpath_db)

# read in init yaml file info for specific run
init_dict = read_inityaml(args.run_name)

# setup pointers to db table files
# TODO: do I want these paths directly from yaml file?
fpath_siteinfo = init_dict['site_summary']
fpath_params = os.path.join(init_dict['path_params'])
dirct_weadata = init_dict['path_wea']
dirct_sims = os.path.join(init_dict['path_sims'])

# fpath_siteinfo = '/home/disk/eos8/ach315/upscale/weadata/site_summary.csv'
# fpath_params = '/home/disk/eos8/ach315/upscale/params/param_opt.csv'
# dirct_weadata = '/home/disk/eos8/ach315/upscale/weadata/data/control/'
# dirct_sims = '/home/disk/eos8/ach315/upscale/sims/opt/'

# insert values into table for specific run
# TODO: is there a way to raise some error if tables have values already?
# TODO: this would signal that instead of make_db I'd want to update_db?
insert_siteinfo(fpath_siteinfo, fpath_db)
insert_params(fpath_params, fpath_db, args.run_name)
insert_weadata(dirct_weadata, fpath_db)
insert_sims(dirct_sims, fpath_db, args.run_name)
