"""Testing wflow setup."""
import os
import numpy as np
import pytest
from numpy import genfromtxt

from ideotype.utils import get_filelist

from ideotype.data import DATA_PATH
from ideotype.wflow_setup import read_yaml, make_dircts, make_runs


run_name = 'test'  # TODO: make sure init_test.yml file is setup correctly
                   # to handle tests!

@pytest.fixtures(scope='module')
def make_tmpath(tmp_path):
    """Create temporary path for workflow testing."""
    dirct_project = tmp_path

    return dirct_project


def test_make_dircts(make_tmpath):
    """Make test directories under temporary path."""
    run_name = 'test'
    dict_setup = read_yaml(run_name)

    # make test directories
    make_dircts(run_name, dict_setup)

    # 




def test_make_runs(make_tmpath):
    # setup project directory for testing as the assigned temp path.
    dirct_project = make_tmpath

    # read in test yaml file with directory setup info
    dict_setup = read_yaml(run_name)
