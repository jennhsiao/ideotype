"""Query sims from database."""

import numpy as np

from ideotype.sql_query import (query_yield,
                                query_phys,
                                query_pheno,
                                query_leaves,
                                query_waterstatus,
                                query_waterpotential)

fpath_db = '/home/disk/eos8/ach315/upscale/db/present.db'
phenos = np.arange(100).tolist()

# Query yield
query, results, df = query_yield(fpath_db, phenos)
# df.to_csv('/home/disk/eos8/ach315/ideotype/sims_present_db.csv', index=False)

# Query physiology
query, results, df = query_phys(fpath_db, phenos)
# df.to_csv('/home/disk/eos8/ach315/ideotype/sims_phys.csv', index=False)

# Query water deficit
query, results, df = query_waterstatus(fpath_db, phenos)
# df.to_csv('/home/disk/eos8/ach315/ideotype/sims_waterdeficit.csv', index=False)

# Query water potential
time = 5
query, results, df = query_waterpotential(fpath_db, phenos, time)
# df.to_csv('/home/disk/eos8/ach315/ideotype/sims_waterpotential.csv', index=False)

# Query phenology
query, results, df = query_pheno(fpath_db, phenos)
# df.to_csv('/home/disk/eos8/ach315/ideotype/sims_phenology.csv', index=False)

# Query morphology
query, results, df = query_leaves(fpath_db, phenos)
# df.to_csv('/home/disk/eos8/ach315/ideotype/sims_leaves.csv', index=False)
