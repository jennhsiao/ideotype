"""Test environment."""
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ideotype.sql_declarative import IdeotypeBase
from ideotype.sql_insert import (insert_siteinfo, insert_params,
                                 insert_sims, insert_weadata)
from ideotype.data import DATA_PATH

import ideotype

test_db = None
# TODO: add test data to /test_data and reference them with relative path
# so you'll have the test files on repo
# os.path.join(DATA_PATH, 'test_data', ...)
dirct_weadata = '/home/disk/eos8/ach315/upscale/db/test_data/wea/*'
dirct_sims = '/home/disk/eos8/ach315/upscale/db/test_data/sim'
fpath_params = '/home/disk/eos8/ach315/upscale/db/test_data/param/param_opt.csv'
fpath_siteinfo = '/home/disk/eos8/ach315/upscale/db/test_data/site/site_summary.csv'


@pytest.fixture(autouse=True, scope='session')
def setup_and_teardown_package():
    """
    Setsup test DB specficially for testing.

    Removes after test is complete.

    """
    global test_engine

    # point to test DB
    fpath_db = '/home/disk/eos8/ach315/upscale/db/test.db'
    # create table function in sql_declarative returns engine
    test_engine = ideotype.create_table(fpath_db)
    # bind metadata with engine
    IdeotypeBase.metadata.bind = test_engine

    # insert table values
    insert_siteinfo(fpath_siteinfo, fpath_db)
    insert_params(fpath_params, fpath_db)
    insert_weadata(dirct_weadata, fpath_db)
    insert_sims(dirct_sims, fpath_db)

    yield test_engine

    # drop all tables once DB tests finish
    IdeotypeBase.metadata.drop_all(test_engine)


@pytest.fixture(scope='function')
def ideotype_session(setup_and_teardown_package):
    """
    Create session connection to test DB.

    Session closed and transactions rolled back after test is complete.

    """
    # takes engine created specific for testing
    test_engine = setup_and_teardown_package
    # connect engine
    test_conn = test_engine.connect()
    # start connection for transaction
    test_trans = test_conn.begin()
    # make a session
    test_session = sessionmaker(bind=test_conn)()
    # TODO: refer back to sql_insert code
    # some differences between sessionmaker vs. an actual session
    # TODO: read up engine, sessions, and DB

    yield test_session

    test_session.close()
    test_trans.rollback()
    test_conn.close()
