"""Reorder primary keys + manually add index."""
from ideotype.sql_altertable import alter_table

fpath_db = '/home/disk/eos8/ach315/upscale/db/ideotype_copy.db'
alter_table(fpath_db)
