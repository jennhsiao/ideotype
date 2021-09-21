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


def query_gseason_climate(fpath_db, phenos):
    """
    Query in-season climate.

    Climate data queried from maizsim output,
    which means there could be slight differences between
    the climate conditions each phenotype experiences
    due to difference in pdate & phenology.

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
    df : pd.DataFrame
        DataFrame of queried results.

    """
    engine = create_engine('sqlite:///' + fpath_db)
    IdeotypeBase.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    query = session.query(Sims.cvar.label('cvar'),
                          Sims.year.label('year'),
                          Sims.site.label('site'),
                          Sims.pheno.label('pheno'),
                          func.avg(Sims.temp_air).label('temp_air'),
                          func.avg(Sims.temp_canopy).label('temp_can'),
                          func.avg(Sims.temp_soil).label('temp_soil'),
                          func.avg(Sims.VPD).label('vpd'),
                          func.avg(Sims.PFD_sun).label('pfd_sun'),
                          func.avg(Sims.PFD_shade).label('pfd_shade'),
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos),
                                              ))
    results = query.all()
    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    return(query, results, df)


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
                          func.avg(Sims.Pn).label('pn'),
                          func.avg(Sims.Pg).label('pg'),
                          func.max(Sims.LAI_sun).label('LAI_sun'),
                          func.max(Sims.LAI_shade).label('LAI_shade'),
                          func.avg(Sims.Ag_sun).label('Ag_sun'),
                          func.avg(Sims.Ag_shade).label('Ag_shade'),
                          func.avg(Sims.An_sun).label('An_sun'),
                          func.avg(Sims.An_shade).label('An_shade')
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
    df['An'] = (df.An_sun * df.sun_perct) + (df.An_shade * df.shade_perct)

    return(query, results, df)


def query_carbon(fpath_db, phenos):
    """
    Query mean and total carbon accumulation across phenostage.

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
                          func.sum(Sims.Pn).label('pn_sum'),
                          func.sum(Sims.Pg).label('pg_sum'),
                          func.avg(Sims.Pn).label('pn_mean'),
                          func.avg(Sims.Pg).label('pg_mean')
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos),
                                              ))

    results = query.all()

    # Construct dataframe from database query
    columns = []
    for item in query.column_descriptions:
        columns.append(item['name'])
    df = pd.DataFrame(results, columns=columns)

    return(query, results, df)


def query_mass(fpath_db, phenos):
    """
    Query mass.

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
                          func.max(Sims.DM_total).label('dm_total'),
                          func.max(Sims.DM_root).label('dm_root'),
                          func.max(Sims.DM_shoot).label('dm_shoot'),
                          func.max(Sims.DM_stem).label('dm_stem'),
                          func.max(Sims.DM_leaf).label('dm_leaf'),
                          func.max(Sims.DM_ear).label('dm_ear'),
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
                          func.count(distinct(Sims.jday)).label('pheno_days'),
                          func.min(Sims.jday).label('jday_start'),
                          func.min(Sims.date).label('date_start')
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
                          func.max(Sims.LA_perplant).label('LA'),
                          func.max(Sims.leaves).label('leaves'),
                          func.max(Sims.leaves_mature).label('leaves_mature'),
                          func.max(Sims.leaves_dropped).label('leaves_dropped')
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
                                  'water_deficit_mean'),
                          func.sum(
                              Sims.ET_sply - Sims.ET_dmd).label(
                                  'water_deficit_sum')
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


def query_waterstatus_sum(fpath_db, phenos):
    """
    Query water status summed across phenostage.

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
                          func.sum(
                              Sims.ET_sply - Sims.ET_dmd).label(
                                  'water_deficit_sum')
                          ).group_by(Sims.cvar,
                                     Sims.year,
                                     Sims.site,
                                     Sims.pheno).filter(
                                         and_(Sims.cvar.in_(phenos),
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
