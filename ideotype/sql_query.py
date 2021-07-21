"""Query DB for analyses."""
import csv

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func, and_
from sqlalchemy.sql.expression import distinct

from ideotype.sql_declarative import (IdeotypeBase,
                                      WeaData,
                                      Sims,
                                      SiteInfo,
                                      Params)


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


def query_yield(fpath_db, phenos):
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

    query = session.query(Sims.cvar.label('cvar'),
                          Sims.year.label('year'),
                          Sims.site.label('site'),
                          Sims.pheno.label('pheno'),
                          func.avg(Sims.DM_ear).label('yield'),
                          SiteInfo.lat.label('lat'),
                          SiteInfo.lon.label('lon'),
                          SiteInfo.texture.label('soil_texture'),
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno,
                                     SiteInfo.site).filter(
                                         and_(Sims.pheno == '"Matured"',
                                              Sims.cvar.in_(phenos),
                                              Sims.site == SiteInfo.site,
                                              Sims.cvar == Params.cvar
                                              ))

    results = query.all()

    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    return(query, results, df)


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
                          ).group_by(Sims.cvar,
                                     Sims.site,
                                     Sims.year,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos)
                                              ))

    results = query.all()

    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    return(query, results, df)


def query_phys(fpath_db, phenos):
    """
    Query phhysiological model outputs during sunlit hours.

    Parameters
    ----------
    fpath_db : str
    phenos : list

    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.cvar.label('cvar'),
                          Sims.site.label('site'),
                          Sims.year.label('year'),
                          Sims.pheno.label('pheno'),
                          func.avg(Sims.av_gs).label('gs'),
                          func.max(Sims.LAI_sun).label('LAI_sun'),
                          func.max(Sims.LAI_shade).label('LAI_shade'),
                          func.avg(Sims.Ag_sun).label('Ag_sun'),
                          func.avg(Sims.Ag_shade).label('Ag_shade')
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos),
                                              Sims.PFD_sun > 0
                                              ))

    results = query.all()

    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    # Scale photosynthesis to canopy
    df['sun_perct'] = df.LAI_sun/(df.LAI_sun + df.LAI_shade)
    df['shade_perct'] = df.LAI_shade/(df.LAI_sun + df.LAI_shade)
    df['Ag'] = (df.Ag_sun * df.sun_perct) + (df.Ag_shade * df.shade_perct)

    return(query, results, df)


def query_leaves(fpath_db, phenos):
    """
    Query physiological model outputs.

    Parameters
    ----------
    fpath_db : str
    phenos : list

    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.cvar.label('cvar'),
                          Sims.site.label('site'),
                          Sims.year.label('year'),
                          Sims.pheno.label('pheno'),
                          func.max(Sims.LAI).label('LAI'),
                          func.max(Sims.LA_perplant).label('LA')
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos)
                                              ))

    results = query.all()

    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    return(query, results, df)


def query_waterstatus(fpath_db, phenos):
    """
    Query water status.

    Parameters
    ----------
    fpath_db : str
    phenos : list

    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.cvar.label('cvar'),
                          Sims.site.label('site'),
                          Sims.year.label('year'),
                          Sims.pheno.label('pheno'),
                          func.avg(
                              Sims.ET_sply - Sims.ET_dmd).label(
                              'water_deficit')
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos),
                                              Sims.time == 12,
                                              ))

    results = query.all()

    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    return(query, results, df)


def query_waterpotential(fpath_db, phenos, time):
    """
    Query water status.

    Parameters
    ----------
    fpath_db : str
    phenos : list
    time : int
        Time to query.

    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.cvar.label('cvar'),
                          Sims.site.label('site'),
                          Sims.year.label('year'),
                          Sims.pheno.label('pheno'),
                          func.avg(Sims.leaf_wp).label('leaf_wp')
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos),
                                              Sims.time == time,
                                              Sims.leaf_wp > -5
                                              ))

    results = query.all()

    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    return(query, results, df)
