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
    print('>> making id_year')
    id_year.create(engine)

    id_site = Index('id_site', Sims.site)
    print('>> making id_site')
    id_site.create(engine)

    id_runame = Index('id_runame', Sims.run_name)
    print('>> making id_runame')
    id_runame.create(engine)

    id_jday = Index('id_jday', Sims.jday)
    print('>> making id_jday')
    id_jday.create(engine)

    id_jday = Index('id_time', Sims.time)
    print('>> making id_time')
    id_jday.create(engine)

    id_pheno = Index('id_pheno', Sims.pheno)
    print('>> making id_pheno')
    id_pheno.create(engine)
