from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sql_declarative import (WeaData, Sims, Params,
                             SiteInfo, Logs, NASSYield, SoilClass)

# TODO: think about sessions - a way to handle conflicts etc.


engine = create_engine('')  # TODO: decide where db is going to live

# Bind engine to metadata of Base class
# so declaratives can be accessed through a DBSession instance
Base.metadata.bind = engine


def insert_weadata(dirct_weadata, session=None):
    """
    Propagate values to DB table WeaData.

    Parameters
    ----------
    dirct_weadata : str
        Directory where all weather file is stored.
        Make sure to include /* at the end in order to fetch
        all files stored in directory.
    session: str
        database session,
    """

    if session is None:
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

    # fetch all weather files in directory
    weafiles = glob.glob(dirct_weadata)

    for weafile in weafiles:
        site_id = weafile.split('/')[-1].split('_')[0]
        year_id = int(weafile.split('/')[-1].split('_')[1].split('.')[0])
        data = genfromtxt(weafile,
                          delimiter='\t',
                          skip_header=1,
                          dtype=(int, object, int, float,
                                 float, float, float, int))

        for row in data:
            # make an object instance
            record = SiteInfo(
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


def insert_sims(session=None):





def insert_sitinfo(fname_siteinfo, session=None):
    """
    insert data into siteinfo table
    """
    if session is None: # can think of this whole session business
                        # as a way to handle conflicts etc.
        DBSession = sessionmaker(bind=engine)
        session = DBSession()
    
    data = genfromtxt(fname_siteinfo,
                      delimiter=',',
                      skip_header=1,
                      dtype=(str, ) # TODO: finish this
                      )
    data = data.tolist()  # maybe don't need to turn to list?
                          # but maybe won't hurt?

    for row in data:
        record = SiteInfo( # making an object instance
            site = row[0],
            class = row[1],
            station = row[2] # TODO: finish this
 
        )
        session.add(record)
    session.commit() # actually adding to DB
        




def update_newrun(): # TODO: this can live elsewhere, maybe?
                     # run to update database
                     # not all tables will need to be updated

