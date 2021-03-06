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
import os
import time
import atexit  # noqa

import numpy as np
import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from numpy import genfromtxt
# from datetime import datetime

from ideotype.sql_declarative import (
    IdeotypeBase, WeaData, Sims, Params,
    SiteInfo, LogInit)
#                             LogMAIZSIM,
#                             NASSYield, SoilClass)

from ideotype.utils import get_filelist, CC_VPD

# code for line profiler
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


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
    Time estimate for DB table inserts.

    Time estimations output to terminal at intervals of numbers of files read.

    Parameters
    ----------
    time_list : list of times at record point.
    count_list : list of file counts at record point.
    nfiles : total number of files left to read in.
    fname : file name for record point.

    """
    time_perloop = (time_list[-1] - time_list[0])/count_list[-1]
    count = count_list[-1]
    estimated_time = (nfiles - count) * time_perloop
    unit = 's'
    if estimated_time > 60:
        estimated_time = estimated_time/60
        unit = 'm'
    if estimated_time > 60:
        estimated_time = estimated_time/60
        unit = 'h'

    print(f'on file {count+1} of {nfiles}, '
          f'filename: {fname}, '
          f'time per loop = {round(time_perloop, 3)} s, '
          f'estimated time remaining {round(estimated_time, 2)} {unit}')


#@profile  # noqa
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
    print('>>> importing weafiles')
    start_time = time.perf_counter()

    if session is None:
        engine = create_engine('sqlite:///' + fpath_db)
        IdeotypeBase.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    # fetch all weather files
    weafiles = get_filelist(dirct_weadata)

    # setup for run-time estimation
    timelist = [time.perf_counter()]
    count_times = [0]
    nfiles = len(weafiles)
    if nfiles > 10:
        printreport_loop = (nfiles*(
            (np.arange(10)+1)/10)).round().astype(int).tolist()
    else:
        printreport_loop = np.arange(nfiles - 1) + 1

    # setup core list
    # following SQLAlchemy Core Method to reduce table insert time:
    # https://docs.sqlalchemy.org/en/13/faq/performance.html#i-m-inserting-400-000-rows-with-the-orm-and-it-s-really-slow
    # create list of dicts that stores table info
    core_list = []

    # loop through individual weather files
    for count, weafile in enumerate(weafiles):
        # estimate run time
        if count == 1 or count in printreport_loop:
            timelist.append(time.perf_counter())
            count_times.append(count)
            fname = weafile.split('/')[-1]
            _time_estimate(timelist, count_times, nfiles, fname)

        # parse info needed from file name
        site_id = weafile.split('/')[-1].split('_')[0]
        year_id = int(weafile.split('/')[-1].split('_')[1].split('.')[0])

        # read individual sim file with pandas
        # pd.read_csv is actually faster than genfromtxt
        data = pd.read_csv(weafile, delimiter='\t')
        # immediately convert dataframe back to recarrays (data.to_records)
        # since accessing individual row info slow in dataframes
        data = data.to_records(index=False)

        for row in data:
            core_list.append(
                {
                    'site': site_id,
                    'year': year_id,
                    'jday': int(row[0]),
                    'date': row[1],
                    'time': int(row[2]),
                    'solar': row[3],
                    'temp': row[4],
                    'precip': row[5],
                    'rh': row[6],
                    'co2': int(row[7]),
                    'vpd': round(CC_VPD(row[4], row[6]/100), 2)
                }
            )

    # insert list of dicts with all weather info into WeaData
    print('> inserting weafiles to DB')
    engine.execute(
        WeaData.__table__.insert(),
        core_list
    )

    # commit data to database
    print('> commiting weafiles to DB')
    session.commit()

    end_time = time.perf_counter()
    print(f'sims total run time: {end_time - start_time} s')


#@profile  # noqa
def insert_sims(dirct_sims, fpath_db, run_name, n_savefiles=100, session=None,
                start_year=None, start_cvar=None, start_site=None):
    """
    Propagate values to DB table - Sims.

    Parameters
    ----------
    dirct_sims : str
        Upper most directory where sim files are stroed.
    fpath_db : str
        Database path.
    run_name : str
        Run name for batch of simulations.
    session : str
        Database session, default to None and generates new session.
    n_savefiles : int
        Number of files to collect before batch inserting into DB.
        Default to 100 but need much smaller value for testing purposes.
    start_year : int
        Specify sim year to start reading from.
        Must also include start_cvar & start_site.
    start_cvar : int
        Specify sim cultivar to start reading from.
        Must also include start_year & start_site.
    start_site : int
        Specify sim site to start reading from.
        Must also include start_year & start_cvar.

    """
    print('>>> importing sims')
    start_time = time.perf_counter()

    if session is None:
        engine = create_engine('sqlite:///' + fpath_db)
        IdeotypeBase.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    # fetch all simulation files
    if (start_year is None) and (start_cvar is None) and (start_site is None):
        # standard read-all case
        simfiles = get_filelist(dirct_sims)
    else:
        # if start point specified for reading in sim files
        dirct_sims_full = os.path.join(dirct_sims,
                                       str(start_year),
                                       'var_' + str(start_cvar),
                                       'out1_' + str(start_site) + '_' +
                                       str(start_year) + '_' +
                                       'var_' + str(start_cvar) + '.txt')
        simfiles = get_filelist(dirct_sims)
        simfile_index = simfiles.index(dirct_sims_full)
        simfiles = simfiles[simfile_index:]

    # setup run name
    runame = run_name

    # setup for run-time estimation
    timelist = [time.perf_counter()]
    count_times = [0]
    nfiles = len(simfiles)
    if nfiles > 1000:
        printreport_loop = (nfiles*(
            (np.arange(1000)+1)/1000)).round().astype(int).tolist()
    else:
        printreport_loop = np.arange(nfiles-1) + 1

    # setup core list - list of dicts
    # following SQLAlchemy Core Method to reduce table insert time:
    core_list = []

    n_to_save = np.ceil(nfiles/n_savefiles).astype(int)
    core_list_iter = list(np.arange(n_to_save)*n_savefiles)
    core_list_iter[-1] = nfiles - 1

    # loop through individual simulation files
    for count, sim in enumerate(simfiles):
        # estimate run time
        if count == 1 or count in printreport_loop:
            timelist.append(time.perf_counter())
            count_times.append(count)
            fname = sim.split('/')[-1]
            _time_estimate(timelist, count_times, nfiles, fname)

        # parse info needed from file name
        site_id = str(sim.split('/')[-1].split('.')[0].split('_')[1])
        year_id = int(sim.split('/')[-1].split('.')[0].split('_')[2])
        cvar_id = int(sim.split('/')[-1].split('.')[0].split('_')[4])

        # read in individual sim file with pandas
        # pd.read_csv better performance over genfromtxt
        data = pd.read_csv(sim, delimiter=',')
        # immediately convert dataframe back to recarrays (data.to_records)
        # for improved process time
        data = data.to_records(index=False)

        # read in each line of sim file
        for row in data:
            core_list.append(
                {
                    'run_name': runame,
                    'site': site_id,
                    'year': year_id,
                    'cvar': cvar_id,
                    'jday': int(row[1]),
                    'time': int(row[2]),
                    'date': row[0],
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

        if count in core_list_iter:
            insert_start = time.perf_counter()
            print(f'> batch sims insert at file #: {count}')

            try:
                engine.execute(
                    Sims.__table__.insert(),
                    core_list
                )
                # commit data to DB
                session.commit()
                insert_end = time.perf_counter()
                insert_time = insert_end - insert_start
                print(f'> insert + commit done in {round((insert_time), 2)} s')
            except IntegrityError:
                print('!!! integrity error - start check !!!')
                for item in core_list:
                    try:
                        engine.execute(
                            Sims.__table__.insert(),
                            item
                        )
                        session.commit()
                    except IntegrityError as error:
                        print(item)
                        raise error

            core_list = []

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
    """Insert maizsim logs."""
    pass


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
