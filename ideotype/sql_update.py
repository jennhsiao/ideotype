from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sql_declarative import (WeaData, Sims, Params,
                             SiteInfo, LogInit, LogMAIZSIM,
                             NASSYield, SoilClass)

from ideotype.sql_insert import (insert_sims,
                                 insert_params,
                                 insert_loginit,
                                 insert_logmaizsim)


# The purpose is to update database if some tables have changed or
# new runs have been generated.
# However, not all tables will need to be updated, and only
# a few selected functions above will need to be run.

def update_newrun():
    """
    """
    pass

