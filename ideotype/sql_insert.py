import glob

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from numpy import genfromtxt

from sql_declarative import (Base, WeaData, Sims, Params,
                             SiteInfo, LogInit, LogMAIZSIM,
                             NASSYield, SoilClass)
from utils import get_filelist

# TODO: think about sessions as SQLAlchemy's
# way to handle conflicts and serelizability, etc.


engine = create_engine('')  # TODO: decide where db is going to live

# Bind engine to metadata of Base class
# so declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine


def insert_weadata(dirct_weadata, session=None):
    """
    Propagate values to DB table - WeaData.

    Parameters
    ----------
    dirct_weadata : str
        Directory where all weather file is stored.
        Make sure to include /* at the end in order to fetch
        all files stored in directory.
    session: str
        Database session, default to None and generates new session.

    """
    if session is None:
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    # fetch all weather files in directory
    # TODO: code up something that will work regardless of
    # directory input format to function
    weafiles = glob.glob(dirct_weadata)

    for weafile in weafiles:
        site_id = weafile.split('/')[-1].split('_')[0]
        year_id = int(weafile.split('/')[-1].split('_')[1].split('.')[0])
        data = genfromtxt(weafile,
                          delimiter='\t',
                          skip_header=1,
                          dtype=(int, 'U12', int, float,
                                 float, float, float, int))

        for row in data:
            # make an object instance
            record = WeaData(
                site=site_id,
                year=year_id,
                jday=row[0],
                date=row[1],
                hour=row[2],
                solrad=row[3],
                temp=row[4],
                precip=row[5],
                rh=row[6],
                co2=row[7])

            # add row data to record
            session.add(record)

        # add data to database
        session.commit()


def insert_sims(dirct_sims, session=None):
    """
    Propagate values to DB table - Sims.

    Parameters
    ----------
    dirct_sims: str
        Upper most directory where sim files are stroed.
    session: str
        Database session, default to None and generates new session.

    """
    if session is None:
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    # fetch all files within directory
    simfiles = get_filelist(dirct_sims)
    runame = dirct_sims.split('/')[-1]

    for sim in simfiles:
        # parse info needed from file name
        site_id = str(sim.split('/')[-1].split('.')[0].split('_')[1])
        year_id = int(sim.split('/')[-1].split('.')[0].split('_')[2])
        cvar_id = int(sim.split('/')[-1].split('.')[0].split('_')[4])

        # read in individual sim file
        data = genfromtxt(
            sim,
            delimiter=',',
            skip_header=1,
            dtype=tuple(['U10'] + [float]*48 + ['<U50']),
            encoding='latin_1'
            )

        # read in each line of sim file
        for row in data:
            record = Sims(
                run_name=runame,
                site=site_id,
                year=year_id,
                cvar=cvar_id,
                date=row[0],
                time=row[2],
                leaves=row[3],
                leaves_mature=row[4],
                leaves_dropped=row[5],
                LA_perplant=row[6],
                LA_dead=row[7],
                LAI=row[8],
                leaf_wp=row[10],
                temp_soil=row[13],
                temp_air=row[14],
                temp_canopy=row[15],
                ET_dmd=row[16],
                ET_sply=row[17],
                Pn=row[18],
                Pg=row[19],
                resp=row[20],
                av_gs=row[21],
                LAI_sun=row[22],
                LAI_shade=row[23],
                PFD_sun=row[24],
                PFD_shade=row[25],
                An_sun=row[26],
                An_shade=row[27],
                Ag_sun=row[28],
                Ag_shade=row[29],
                gs_sun=row[30],
                gs_shade=row[31],
                VPD=row[32],
                Nitr=row[33],
                N_Dem=row[34],
                NUpt=row[35],
                LeafN=row[36],
                PCRL=row[37],
                DM_total=row[38],
                DM_shoot=row[39],
                DM_ear=row[40],
                DM_leaf=row[41],
                DM_stem=row[43],
                DM_root=row[44],
                AvailW=row[47],
                solubleC=row[48],
                Pheno=row[49].srip()  # remove leading whitespace in output
            )

            # add row data to record
            session.add(record)

        # commit data to DB
        session.commit()


def insert_params(fpath_params, session=None):
    """
    Propagate values to DB table - Params.

    Parameters
    ----------
    fpath_params: str
        Path to params file.
    session: str
        Database session, default to None and generates new session.

    """
    if session is None:
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
    runame = fpath_params.split('/')[-1].split('.')[0].split('_')[1]
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


# TODO: need to modify this site_summary csv file
# to not include an index column
def insert_sitinfo(fpath_siteinfo, session=None):
    """
    Propagate values to DB table - SiteInfo.

    Parameters
    ----------
    fpath_siteinfo : str
        Path to site_info file.
    session: str
        Database session, default to None and generates new session.

    """
    if session is None:
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    data = genfromtxt(
        fpath_siteinfo,
        delimiter=',',
        skip_header=1,
        dtype=(int, 'U6', int, '<U40', 'U2',
               int, float, float, int, float, float),
        )

    for row in data:
        # make an object instance of the SiteInfo table
        record = SiteInfo(
            site=row[0],
            state=row[3],
            lat=row[5],
            lon=row[6],
            years=row[7],
            area=row[8],
            perct_irri=row[9],
        )

        # add row data to record
        session.add(record)

    # commit data to DB
    session.commit()


def insert_loginit(fpath_log, session=None):
    """
    Propagate values to DB table - LogInit.

    Parameters
    ----------
    fpath_log: str
        File path for experiment log file.
    session:
        Database session, default to None and generates new session.

    """
    if session is None:
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


def insert_all():
    """
    Combines individual insert_table functions create DB.

    """
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
