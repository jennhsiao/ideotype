"""
Debug error when inserting simulation outputs into DB.

- Leveraging options in insert_sims to batch save records to DB
at a smaller number, making it easier to debug errors.
- Also starting import from specified year & cultivar
to narrow down search range.

"""
from ideotype import insert_sims

fpath_db = '/home/disk/eos8/ach315/upscale/db/ideotype.db'
dirct_weadata = '/home/disk/eos8/ach315/upscale/weadata/data/control/'
dirct_sims = '/home/disk/eos8/ach315/upscale/sims/opt/'
fpath_params = '/home/disk/eos8/ach315/upscale/params/param_opt.csv'
fpath_siteinfo = '/home/disk/eos8/ach315/upscale/weadata/site_summary.csv'
run_name = 'opt'

insert_sims(dirct_sims, fpath_db, run_name,
            n_savefiles=100, session=None,
            start_year=1994, start_cvar=94, start_site=722060)

# multiple files in 1994/var_93 have issues
# order messed up or duplicated rows (don't know how this happened)
# skipped all the sites in this directory at the moment
