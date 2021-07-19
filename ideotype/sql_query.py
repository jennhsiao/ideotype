"""Query DB for analyses."""
import csv
from re import M

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, and_
from sqlalchemy.sql.expression import distinct

from ideotype.sql_declarative import (
    IdeotypeBase, WeaData, Sims, Params,
    SiteInfo, LogInit)


def query_weadata(fpath_db):
    """
    Weathere data query.

    - Average meteorology at each site.
    - Variance of meteorology at each site.

    """
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

    #  query output as csv
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])

    with open('testoutput.csv', 'w') as outfile:
        outcsv = csv.writer(outfile)
        outcsv.writerow(columns)
        for row in results:
            outcsv.writerow(row)


def query_sims(fpath_db):
    """
    Sims query.

    - Final yield for each site-year-cvar combination.
    - Yield variation across cvars.
    - Yield variation across sites.
    - Yield variation across years.

    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.site.label('site'),
                          Sims.year.label('year'),
                          Sims.cvar.label('cavr'),
                          Sims.DM_ear.label('DM_ear'),
                          Sims.pheno.label('pheno')
                          ).group_by(Sims.site, Sims.cvar).filter(
                              Sims.pheno == '"Matured"')
    results = query.all()

    return(results)

#    query = session.query(Sims.pheno).group_by(Sims.pheno).distinct()

#    query = session.query(Sims.site.label('site'),
#                          Sims.cvar.label('cavr'),
#                          func.avg(Sims.DM_ear).label('mean_DM_ear'),
#                          func.stdev(Sims.DM_ear).label('stdev_DM_ear'),
#                          Sims.pheno.label('pheno')
#                          ).group_by(Sims.site, Sims.cvar).filter(
#                              Sims.pheno == '"grainFill"')


def query_pheno(fpath_db, phenos):
    """
    Query pheno info.

    Parameters
    ----------
    fpath_db : str
    phenos : list
        List of top phenotype numbers.

    Returns
    -------
    query : sqlalchemy query
    results : list
        List of query results.

    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.cvar.label('cvar'),
                          Sims.site.label('site'),
                          Sims.year.label('year'),
                          Sims.pheno.label('pheno'),
                          func.count(distinct(Sims.jday)).label('pheno_days')
                          ).group_by(
                              Sims.cvar, Sims.site, Sims.year, Sims.pheno
                          ).filter(
                              and_(
                                  Sims.cvar.in_(phenos)
                              ))

    results = query.all()

    return(query, results)
