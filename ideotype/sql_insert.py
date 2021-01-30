"""
Insert table values into database.

Tables include:
- WeaData
- Sims
- Params
- SiteInfo
- LogInit

* Note:
* Order of insert table need to depend on
* foreign key construction.
* Foreign keys can't be inserted prior to their
* linked primay keys.

"""
import time
import atexit

import numpy as np
import pandas as pd
import line_profiler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from numpy import genfromtxt
from datetime import datetime

from ideotype.sql_declarative import (
    IdeotypeBase, WeaData, Sims, Params,
    SiteInfo, LogInit)
#                             LogMAIZSIM,
#                             NASSYield, SoilClass)

from ideotype.utils import get_filelist, CC_VPD

profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


def insert_siteinfo(fpath_siteinfo, fpath_db, session=None):
    """
    Propagate values to DB table - SiteInfo.

    Parameters
    ----------
    fpath_siteinfo : str
        Path to site_info file.
    fpath_db: str
        Database path.
    session: str
        Database session, default to None and generates new session.

    """
    if session is None:
        engine = create_engine('sqlite:///' + fpath_db)
        IdeotypeBase.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    data = genfromtxt(
        fpath_siteinfo,
        delimiter=',',
        skip_header=1,
        dtype=('U6', int, '<U40', 'U2', int,
               float, float, int, float, float),
        )

    for row in data:
        # make an object instance of the SiteInfo table
        record = SiteInfo(
            site=(row[0]),
            state=row[3],
            lat=row[5],
            lon=row[6],
            years=int(row[7]),
            area=round(row[8], 2),
            perct_irri=round(row[9], 2),
        )

        # add row data to record
        session.add(record)

    # commit data to DB
    session.commit()


def insert_params(fpath_params, fpath_db, run_name, session=None):
    """
    Propagate values to DB table - Params.

    Parameters
    ----------
    fpath_params: str
        Path to params file.
    fpath_db: str
        Database path.
    run_name: str
        Run name for batch of simulations.
    session: str
        Database session, default to None and generates new session.

    """
    if session is None:
        engine = create_engine('sqlite:///' + fpath_db)
        IdeotypeBase.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    data = genfromtxt(
        fpath_params,
        delimiter=',',
        names=True,
        dtype=(float)
        )

    # parse out run_name from parameter file name
    # TODO: need to make sure parameter file name is generated systematically
    # TODO: is there a way to write a test to always check for this?
    runame = run_name
    # fetch parameters
    params = data.dtype.names

    for count, row in enumerate(data):
        for par in params:
            # make an object instance of the Params table
            record = Params(
                run_name=runame,
                cvar=count,
                param=par,
                value=row[par]
            )

            # add row data to record
            session.add(record)

    # commit data to DB
    session.commit()


def _time_estimate(time_list, count_list, nfiles, fname):
    """
    """
    time_perloop = (time_list[-1] - time_list[0])/count_list[-1]
    count = count_list[-1]
    estimated_time = (nfiles - count)/time_perloop
    unit = 's'
    if estimated_time > 60:
        estimated_time = estimated_time/60
        unit = 'm'
    if estimated_time > 60:
        estimated_time = estimated_time/60
        unit = 'h'

    print(f'on file {count+1} of {nfiles}, '
          f'filename: {fname}, '
          f'time per loop = {time_perloop} s, '
          f'estimated time remaining {estimated_time} {unit}')


#@profile
def insert_weadata(dirct_weadata, fpath_db, session=None):
    """
    Propagate values to DB table - WeaData.

    Parameters
    ----------
    dirct_weadata : str
        Directory where all weather file is stored.
        Make sure to include /* at the end in order to fetch
        all files stored in directory.
    fpath_db: str
        Database path.
    session: str
        Database session, default to None and generates new session.

    """
    print('importing weafiles')
    if session is None:
        engine = create_engine('sqlite:///' + fpath_db)
        IdeotypeBase.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    weafiles = get_filelist(dirct_weadata)

    timelist = [time.perf_counter()]
    count_times = [0]
    nfiles = len(weafiles)
    if nfiles > 10:
        printreport_loop = (nfiles*(
            (np.arange(10)+1)/10)).round().astype(int).tolist()
    else:
        printreport_loop = np.arange(nfiles - 1) + 1

    for count, weafile in enumerate(weafiles):
        if count == 1 or count in printreport_loop:
            timelist.append(time.perf_counter())
            count_times.append(count)
            _time_estimate(timelist, count_times, nfiles, weafile)

        site_id = weafile.split('/')[-1].split('_')[0]
        year_id = int(weafile.split('/')[-1].split('_')[1].split('.')[0])
        data = genfromtxt(weafile,
                          delimiter='\t',
                          skip_header=1,
                          dtype=(int, 'U12', int, float,
                                 float, float, float, int))

        object_list = []
        for row in data:
            # make an object instance
            record = WeaData(
                site=site_id,
                year=year_id,
                jday=int(row[0]),
                date=datetime.strptime(row[1].strip("'"), '%m/%d/%Y').date(),
                time=int(row[2]),
                solar=row[3],
                temp=row[4],
                precip=row[5],
                rh=row[6],
                co2=int(row[7]),
                vpd=round(CC_VPD(row[4], row[6]/100), 2))

            # add row data to record
            #session.add(record)
            object_list.append(record)

        session.bulk_save_objects(object_list)
        # add data to database
        session.commit()


#@profile
def insert_sims(dirct_sims, fpath_db, run_name, session=None, core=True):
    """
    Propagate values to DB table - Sims.

    Parameters
    ----------
    dirct_sims: str
        Upper most directory where sim files are stroed.
    fpath_db: str
        Database path.
    run_name: str
        Run name for batch of simulations.
    session: str
        Database session, default to None and generates new session.

    """
    print('importing sims')
    start_time = time.perf_counter()
    if session is None:
        engine = create_engine('sqlite:///' + fpath_db)
        IdeotypeBase.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    simfiles = get_filelist(dirct_sims)
    runame = run_name

    timelist = [time.perf_counter()]
    count_times = [0]
    nfiles = len(simfiles)

    if nfiles > 10:
        printreport_loop = (nfiles*(
            (np.arange(10)+1)/10)).round().astype(int).tolist()
    else:
        printreport_loop = np.arange(nfiles-1) + 1

    object_list = []
    core_list = []

    for count, sim in enumerate(simfiles):
        if count == 1 or count in printreport_loop:
            timelist.append(time.perf_counter())
            count_times.append(count)
            _time_estimate(timelist, count_times, nfiles, sim)

        # parse info needed from file name
        site_id = str(sim.split('/')[-1].split('.')[0].split('_')[1])
        year_id = int(sim.split('/')[-1].split('.')[0].split('_')[2])
        cvar_id = int(sim.split('/')[-1].split('.')[0].split('_')[4])

        # read in individual sim file
#        data = genfromtxt(
#            sim,
#            delimiter=',',
#            skip_header=1,
#            dtype=tuple(['U10'] + [float]*48 + ['<U50']),
#            )

        # pandas read_csv actually faster than genfromtxt
        data = pd.read_csv(sim, delimiter=',')
        data = data.to_records(index=False)

        # read in each line of sim file
        #object_list = []
        for row in data:
            core_list.append(
                {
                    'run_name': runame,
                    'site': site_id,
                    'year': year_id,
                    'cvar': cvar_id,
                    'jday': row[1],
                    'date': row[0],
                    'time': row[2],
                    'leaves': row[3],
                    'leaves_mature': row[4],
                    'leaves_dropped': row[5],
                    'LA_perplant': row[6],
                    'LA_dead': row[7],
                    'LAI': row[8],
                    'leaf_wp': row[10],
                    'temp_soil': row[13],
                    'temp_air': row[14],
                    'temp_canopy': row[15],
                    'ET_dmd': row[16],
                    'ET_sply': row[17],
                    'Pn': row[18],
                    'Pg': row[19],
                    'resp': row[20],
                    'av_gs': row[21],
                    'LAI_sun': row[22],
                    'LAI_shade': row[23],
                    'PFD_sun': row[24],
                    'PFD_shade': row[25],
                    'An_sun': row[26],
                    'An_shade': row[27],
                    'Ag_sun': row[28],
                    'Ag_shade': row[29],
                    'gs_sun': row[30],
                    'gs_shade': row[31],
                    'VPD': row[32],
                    'Nitr': row[33],
                    'N_Dem': row[34],
                    'NUpt': row[35],
                    'LeafN': row[36],
                    'PCRL': row[37],
                    'DM_total': row[38],
                    'DM_shoot': row[39],
                    'DM_ear': row[40],
                    'DM_leaf': row[41],
                    'DM_stem': row[43],
                    'DM_root': row[44],
                    'AvailW': row[47],
                    'solubleC': row[48],
                    'pheno': row[49].strip()
                }

            )

    engine.execute(
        Sims.__table__.insert(),
        core_list
    )

    # commit data to DB
    session.commit()

    end_time = time.perf_counter()
    print(f'sims total run time: {end_time - start_time} s')


def insert_loginit(fpath_log, fpath_db, session=None):
    """
    Propagate values to DB table - LogInit.

    Parameters
    ----------
    fpath_log: str
        File path for experiment log file.
    fpath_db: str
        Database path.
    session:
        Database session, default to None and generates new session.

    """
    if session is None:
        engine = create_engine('sqlite:///' + fpath_db)
        IdeotypeBase.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    data = genfromtxt(
        fpath_log,
        delimiter=',',
        skip_header=1,
        dtype=(str)
        )

    for row in data:
        record = LogInit(
            run_name=row[0],
            init_yml=row[1],
            path_inits=row[2],
            path_params=row[3],
            path_jobs=row[4],
            path_sims=row[5],
            path_maizsim=row[6],
            siteyears=row[7],
            site_info=row[8],
            site_summary=row[9],
            pdate=row[10],
            version=row[11]
        )

        # add row data to record
        session.add(record)

    # commit data to DB
    session.commit()


def insert_logmaizsim():
    """
    """
    pass



# TODO: split into two funcions
# 1. things updated per run
# 2. one off things to import
def insert_all():
    """
    Combines individual insert_table functions create DB.

    Single-time execution.

    """
    pass
    # TODO: call function input parameters from yaml file
    # most of the function parameters need file paths or directories
    # to get to the data needed to build the database

    # WeaData

    # Sims

    # Params

    # SiteInfo

    # LogInit

    # LogMAIZSIM

    # NASSYield

    # SoilClass


def insert_update():
    """
    Insert value to tables that will need updates after new experiments.

    Execute after new batch of experiments.
    Tables that will need update:
    - sims
    - params
    - log_init
    - log_maizsim

    """
    pass
