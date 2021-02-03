"""Query DB for analyses."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func

from ideotype.sql_declarative import (
    IdeotypeBase, WeaData, Sims, Params,
    SiteInfo, LogInit)


def query_weadata(fpath_db):
    """
    Weathere data query.

    - Average meteorology at each site.
    - Variance of meteorology at each site.

    """
    # TODO: figure out how to query mean total rainfall across years per site
    # TODO: calculate variance?

    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(WeaData.site.label('site'),
                          func.avg(WeaData.temp).label('mean_temp'),
                          func.avg(WeaData.vpd).label('mean_vpd'),
                          func.sum(WeaData.precip).label('total_precip'),
                          func.count(WeaData.precip).label('precip_count')
                          ).group_by(WeaData.site)
    results = query.all()



def query_sims(fpath_db):
    """
    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.site.label('site'),
                          Sims.cvar.label('cavr'),
                          Sims.DM_ear.label('DM_ear'),
                          Sims.pheno.label('pheno')
                          ).group_by(Sims.site, Sims.cvar).filter(
                              Sims.pheno == '"Matured"'
                          )
