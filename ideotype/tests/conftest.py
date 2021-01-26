"""Test environment."""
import pytest

from sqlalchemy.orm import sessionmaker

import ideotype

test_db = None


@pytest.fixture(autouse=True, scope='session')
def setup_and_teardown_package():
    """
    Setsup test DB specficially for testing.

    Removes after test is complete.

    """
    global test_db

    # point to test DB
    fpath_db = '/home/disk/eos8/ach315/upscale/db/test.db'

    test_db = ideotype.create_table(fpath_db)

    yield test_db

    test_db.drop_tables()


@pytest.fixtures(scope='function')
def ideotype_session(setup_and_teardown_package):
    """
    Create session connection to test DB.

    Session closed and transactions rolled back after test is complete.

    """
    test_db = setup_and_teardown_package
    test_conn = test_db.engine.connect()
    test_trans = test_conn.begin()  # transaction
    test_session = sessionmaker(bind=test_conn)

    yield test_session

    test_session.close()
    test_trans.rollback()
    test_conn.close()
