import os
import sys

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

# TODO: think about how foreign keys should work here
# specifically, joint foresign keys
# TODO: check code Bryna sent in Slack

Base = declarative_base()  # declarative_base is how you define tables


class WeaData(Base):
    __tablename__ = 'weadata'
    site = Column(String(6), primary_key=True)
    year = Column(Integer, primary_key=True)
    jday = Column(Integer)
    date = Column(DateTime(), primary_key=True)  # TODO: check format
    time = Column(Integer, primary_key=True)
    solar = Column(Float)
    temp = Column(Float)
    precip = Column(Float)
    rh = Column(Float)
    co2 = Column(Integer)
    vpd = Columnn(Float)


class Sims(Base):
    __tablename__ = 'sims'
    run_name = Column(String, primary_key=True)
    site = Column(String, primary_key=True)
    year = Column()
    cvar =      
        # TODO: what if cvar scheme changed between runs?
        # this would definitely lead to errors if I tried to
        # query and compare cultivars between runs.
        # However, I guess it's maybe okay that the exact sampled
        # parameter values are not the same, as long as I can access
        # those values through the Params table?
        # TODO: address in Params table
    
    date = 
    time = 
    
    # below are not primary keys or foreign keys
    leaves = 
    leaves_mature = 
    leaves_dropped = 
    LA_perplant = 
    LA_dead = 
    LAI = 
    leaf_wp = 
    temp_soil = 
    temp_air = 
    temp_canopy = 
    ET_dmd = 
    ET_sply = 
    Pn = 
    Pg = 
    resp = 
    av_gs = 
    LAI_sun = 
    LAI_shade = 
    PFD_sun = 
    PFD_shade = 
    An_sun = 
    An_shade = 
    Ag_sun = 
    Ag_shade = 
    gs_sun = 
    gs_shade = 
    # TODO: so much more to add


class Params(Base):
    __tablename__ = 'params'
    # TODO: add another column - params
    # included as part of primary key


class SiteInfo(Base):
    # TODO: add site as foreign key
    # TODO: look up foreign key & relationship
    # TODO: not sure why/when to use relationships as shown in doc below
    # TODO: https://docs.sqlalchemy.org/en/13/orm/basic_relationships.html

    __tablename__ = 'siteinfo'
    site = Column(String(6),
                  primary_key=True,
                  ForeignKey('Sims.site')),
    state = Column(String(2)),
    lat = Column(Float),
    lon = Column(Float),
    years = Column(Integer),
    area = Column(Float)
    perct_irri = Column(Float)



class LogInit(Base):
    # TODO: not sure how to grab logs for the maizsim githash
    # TODO: should be it's own table
    __tablename__ = 'log_init'
    run_name = Column(String, ForeignKey(''))


class LogMAIZSIM(Base):
    __tablename__ = 'log_maizsim'


class NASSYield(Base):
    __tablename__ = 'nass_yield'
    # TODO: this is an issue here since lat/lon here
    # don't actually correspond to lat/lon in site
    # I use lat/lon to calcualte the nearest site/sites
    # and calculate mean [yield/planting area/irrigation for comparison
    # TODO: probably a good way to go about is to create a relations
    # table that links the two tables
    # this can be a one-to-many relation


class SoilClass(Base):
    __tablename__ = 'soil_class'
    site = Column(String, primary_key=True)
    # TODO: similar issue
    # linked to NASS_yield but not to the rest of the DB


# create engine to setup database
engine = create_engine('')

# create all tables in engine
# = 'Create Table' in SQL
Base.metadata.create_all(engine)
