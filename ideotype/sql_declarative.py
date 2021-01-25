"""
Set up and create tables for SQL database.

"""
from sqlalchemy import (Column, ForeignKey, ForeignKeyConstraint,
                        Integer, String, Float, DateTime)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine


class IdeotypeBase(object):
    """
    Add specifics to Base to debug.

    __repr__:

    """

    def __repr__(self):  # magic function __repr__
        """Define standard representation."""
        columns = self.__table__.columns.keys()
        rep_str = '<' + self.__class__.__name__ + '('
        for c in columns:
            rep_str += str(getattr(self, c)) + ', '  # getattr
        rep_str = rep_str[0:-2]
        rep_str += ')>'

        return rep_str


# declarative_base is how you define tables
# makes an instance of a declarative_base() object
IdeotypeBase = declarative_base(cls=IdeotypeBase)


class WeaData(IdeotypeBase):
    """
    DB table for weather data table.

    Parameters
    ----------
    site: String Column
        Simulation site.

    """

    __tablename__ = 'weadata'
    site = Column(String(6),
                  ForeignKey('site_info.site'),
                  primary_key=True)
    year = Column(Integer, primary_key=True)
    jday = Column(Integer)
    date = Column(DateTime, primary_key=True)
    time = Column(Integer, primary_key=True)
    solar = Column(Float)
    temp = Column(Float)
    precip = Column(Float)
    rh = Column(Float)
    co2 = Column(Integer)
    vpd = Column(Float)


class Sims(IdeotypeBase):
    """
    DB table for simulation outputs.

    Attributes
    ----------
    run_name: String Column
        Run name for bath of simulation experiments. Part of primary_key.
    cvar: Integer Column
        Cultivar number that represents specific param combinations.
    site: String Column
        Simulation site id.
    year: Integer Column
        Simualtion year.
    cvar: String Column
        Simulation cultivar choice.
    date: DateTime Column
        MAIZSIM simulation output timestep date.
    time: Integer Column
        MAIZSIM simulation output timestep hour.

    """

    __tablename__ = 'sims'
    # primary keys
    # TODO: need a foreign key constrain here for
    # TODO: run_name & cvar
    run_name = Column(String(20),
                      ForeignKey('params.run_name'),
                      primary_key=True)
    site = Column(String(6),
                  ForeignKey('site_info.site'),
                  primary_key=True)
    year = Column(Integer, primary_key=True)
    cvar = Column(Integer,
                  ForeignKey('params.cvar'),
                  primary_key=True)
    date = Column(DateTime, primary_key=True)
    time = Column(Integer, primary_key=True)

    __table_args__ = (ForeignKeyConstraint(
        ['run_name', 'cvar'], ['params.run_name', 'params.cvar']), {})

    # other columns
    leaves = Column(Float)
    leaves_mature = Column(Float)
    leaves_dropped = Column(Float)
    LA_perplant = Column(Float)
    LA_dead = Column(Float)
    LAI = Column(Float)
    leaf_wp = Column(Float)
    temp_soil = Column(Float)
    temp_air = Column(Float)
    temp_canopy = Column(Float)
    ET_dmd = Column(Float)
    ET_sply = Column(Float)
    Pn = Column(Float)
    Pg = Column(Float)
    resp = Column(Float)
    av_gs = Column(Float)
    LAI_sun = Column(Float)
    LAI_shade = Column(Float)
    PFD_sun = Column(Float)
    PFD_shade = Column(Float)
    An_sun = Column(Float)
    An_shade = Column(Float)
    Ag_sun = Column(Float)
    Ag_shade = Column(Float)
    gs_sun = Column(Float)
    gs_shade = Column(Float)
    VPD = Column(Float)
    Nitr = Column(Float)
    N_Dem = Column(Float)
    NUpt = Column(Float)
    LeafN = Column(Float)
    PCRL = Column(Float)
    DM_total = Column(Float)
    DM_shoot = Column(Float)
    DM_ear = Column(Float)
    DM_leaf = Column(Float)
    DM_stem = Column(Float)
    DM_root = Column(Float)
    AvailW = Column(Float)
    solubleC = Column(Float)
    pheno = Column(String)


class Params(IdeotypeBase):
    """
    DB table for sampled parameter combinations.

    Attributes
    ----------
    run_name: String Column
        Run name of simulation experiments. Part of primary key.
    cvar: Integer Column
        Cultivar number that represents specific param combinations.
    param: String Column
        Perturbed parameter.
    value: Float Column
        Parameter value.

    """

    __tablename__ = 'params'
    # primary keys
    # TODO: want run_name to be primay key here
    # have other tables reference to it via foreignkey
    run_name = Column(String(20), primary_key=True)
    cvar = Column(Integer, primary_key=True)
    param = Column(String, primary_key=True)
    value = Column(Float)


class SiteInfo(IdeotypeBase):
    """
    DB table for simulation site info.

    Attributes
    ----------
    site: String Column
        Simulation site. Primary key.
    state: String Column
    lat: Float Column
    lon: Float Column
    years: Integer Column
        Years of weather data available at simulation site.
    area: Float Column
        Area maize planted (#TODO: find unit).
        Average value from nearby NASS sites (#TODO: find how many site).
    perct_irri: Float Column
        Percent irrigated for simulation site.
        Average value from nearby NASS sites.

    """

    __tablename__ = 'site_info'
    site = Column(String(6), primary_key=True)
    state = Column(String(2))
    lat = Column(Float)
    lon = Column(Float)
    years = Column(Integer)
    area = Column(Float)
    perct_irri = Column(Float)


class LogInit(IdeotypeBase):
    """
    DB table for log files.

    Attributes
    ----------
    run_name: String Column
        Run name of simulation experiments.
        Primary key.
        Foreign key link to Sims and Params table.
    init_yml: String Column
        init yaml file used for experiment.
    path_inits: String Column
        Path where inits are stored for experiment.
    path_params: String Column
        Path where inits are stored for experiment.
    path_jobs: String Column
        Path where jobs are stored for experiment.
    path_sims: String Column
        Path where sims are stroed for experiment.
    path_maizsim: String Column
        Path pointing to maizsim directory used.
    siteyears: String Column
        Path pointing to siteyears efile used.
    site_info: String Column
        Path pointing to site_info file used.
    site_summary: String Column
        Path pointing to site_summary file used.
    pdate: String Column
        Planting date set for simualtions.
    version: String Column
        ideotype version - git hash.

    """

    __tablename__ = 'log_init'
    run_name = Column(String,
                      ForeignKey('params.run_name'),
                      primary_key=True)
    init_yml = Column(String)
    path_inits = Column(String)
    path_params = Column(String)
    path_jobs = Column(String)
    path_sims = Column(String)
    path_maizsim = Column(String)
    siteyears = Column(String)
    site_info = Column(String)
    site_summary = Column(String)
    pdate = Column(String)
    version = Column(String)


#class LogMAIZSIM(Base):
#    # TODO: not sure how to capture this yet
#    __tablename__ = 'log_maizsim'


#class NASSYield(Base):
#    __tablename__ = 'nass_yield'
    # TODO: this is an issue here since lat/lon here
    # don't actually correspond to lat/lon in site
    # I use lat/lon to calcualte the nearest site/sites
    # and calculate mean [yield/planting area/irrigation for comparison
    # TODO: probably a good way to go about is to create a relations
    # table that links the two tables
    # this can be a one-to-many relation


#class SoilClass(Base):
#    __tablename__ = 'soil_class'
#    site = Column(String, primary_key=True)
    # TODO: similar issue
    # linked to NASS_yield but not to the rest of the DB


# TODO: maybe move elsewhere - consider later

def create_table(fpath_db):
    """
    Create table in engine.

    Parameters
    ----------
    fpath_db: str
        Path pointing to database.

    """
    # create engine to setup database
    engine = create_engine('sqlite:///' + fpath_db)

    # create all tables in engine
    # = 'Create Table' in SQL
#    Base.metadata.create_all(engine)
    IdeotypeBase.metadata.create_all(engine)
