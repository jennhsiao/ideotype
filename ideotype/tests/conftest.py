"""Test environment."""
import os

import pytest
from sqlalchemy.orm import sessionmaker
from ideotype.sql_declarative import IdeotypeBase
from ideotype.sql_insert import (insert_siteinfo, insert_params,
                                 insert_sims, insert_weadata)

import ideotype
from ideotype.data import DATA_PATH

# setup test database
test_db = None

# setup paths pointing to data for test DB tables
fpath_siteinfo = os.path.join(DATA_PATH, 'test_data', 'sites',
                              'site_summary.csv')
fpath_params = os.path.join(DATA_PATH, 'test_data', 'params', 'param_test.csv')
dirct_weadata = os.path.join(DATA_PATH, 'test_data', 'wea')
dirct_sims = os.path.join(DATA_PATH, 'test_data', 'sims')


@pytest.fixture(autouse=True, scope='session')
def setup_and_teardown_package():
    """
    Setsup test DB specficially for testing.

    Removes after test is complete.

    """
    global test_engine

    # point to test DB
    fpath_db = os.path.expanduser('~/upscale/db/test.db')
    # create table function in sql_declarative returns engine
    test_engine = ideotype.create_table(fpath_db)
    # bind metadata with engine
    IdeotypeBase.metadata.bind = test_engine

    # setup run name
    run_name = 'test'

    # insert table values
    insert_siteinfo(fpath_siteinfo, fpath_db)
    insert_params(fpath_params, fpath_db, run_name)
    insert_weadata(dirct_weadata, fpath_db)
    insert_sims(dirct_sims, fpath_db, run_name, n_savefiles=3)

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

    yield test_session

    test_session.close()
    test_trans.rollback()
    test_conn.close()
