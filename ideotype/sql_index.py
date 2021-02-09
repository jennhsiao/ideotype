"""Make add-hoc indexes for DB."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Index
from ideotype.sql_declarative import Sims


def make_index(fpath_db):
    """Create index for tables that already exist."""
    engine = create_engine('sqlite:///' + fpath_db)

    id_year = Index('id_year', Sims.year)
    id_year.create(engine)

    id_site = Index('id_site', Sims.site)
    id_site.create(engine)

    id_cvar = Index('id_cvar', Sims.cvar)
    id_cvar.create(engine)

    id_pheno = Index('id_pheno', Sims.pheno)
    id_pheno.create(engine)
