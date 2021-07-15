"""Make add-hoc indexes for DB."""
from sqlalchemy import create_engine
from sqlalchemy import Index
from ideotype.sql_declarative import Sims


def make_index(fpath_db):
    """
    Create index for tables that already exist.

    Add additional indexes other than pre-determined primary keys
    to speed up queries.

    """
    engine = create_engine('sqlite:///' + fpath_db)

    id_year = Index('id_year', Sims.year)
    id_year.create(engine)

    id_site = Index('id_site', Sims.site)
    id_site.create(engine)

    id_runame = Index('id_runame', Sims.run_name)
    id_runame.create(engine)

    id_jday = Index('id_jday', Sims.jday)
    id_jday.create(engine)

    id_pheno = Index('id_pheno', Sims.pheno)
    id_pheno.create(engine)
