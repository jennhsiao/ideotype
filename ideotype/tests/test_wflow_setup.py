"""Testing wflow setup."""

import os
import pytest
import yaml
from shutil import copyfile

from ideotype.utils import get_filelist
from ideotype.data import DATA_PATH
from ideotype.wflow_setup import (make_dircts, make_runs,
                                  make_jobs, make_subjobs)


@pytest.fixture(scope='module')
def make_testyaml(tmp_path_factory):
    """Update init_test.yml file on the fly."""
    # link to init_test.yml
    tmp_path = tmp_path_factory.mktemp('ideotype_test')

    # TODO: will need to make standard directories in tmp_path
    # that mirror how things look like in the standard project directory
    # TODO: shutil (standard library for shell commands, copyfile)
    # use this to copy the init files you need into the tmp_paths

    # create the four main directories under the temp project directory
    os.mkdir(os.path.join(tmp_path, 'inits'))
    os.mkdir(os.path.join(tmp_path, 'jobs'))
    os.mkdir(os.path.join(tmp_path, 'runs'))
    os.mkdir(os.path.join(tmp_path, 'sims'))

    # create secondary directories that need to exist
    os.mkdir(os.path.join(tmp_path, 'inits', 'standards'))
    os.mkdir(os.path.join(tmp_path, 'inits', 'soils'))
    os.mkdir(os.path.join(tmp_path, 'inits', 'cultivars'))
    os.mkdir(os.path.join(tmp_path, 'inits', 'customs'))

    # copy and move files needed for testing purposes that
    # exist in normal project dirct


    # create test_yaml file for testing purposes
    test_yaml = os.path.join(DATA_PATH, 'inits', 'init_test.yml')
    updated_yaml = os.path.join(tmp_path, 'init_test.yml')

    # check if init_test.yml exists
    if not os.path.exists(test_yaml):
        raise ValueError(f'{test_yaml} does not exist!')

    # read in init_test.yml
    with open(test_yaml, 'r') as pfile:
        dict_init = yaml.safe_load(pfile)

    # insert tmp_path as path_project
    dict_init['setup']['path_project'] = str(tmp_path)

    # overwrite existing init_test.yml file
    with open(updated_yaml, 'w') as outfile:
        yaml.dump(dict_init, outfile,
                  default_flow_style=False, sort_keys=False)

    return updated_yaml


#def test_tmpath(tmp_path):


def test_make_dircts(make_testyaml):
    """Make test directories under temporary path."""
    run_name = 'test'
    yamlfile = make_testyaml
    make_dircts(run_name,
                yamlfile=yamlfile,
                cont_years=False,
                cont_cvars=False)  # make test directories

    # code to test directories are correct
    get_filelist()


def test_make_runs(make_testyaml):
    run_name = 'test'
    yamlfile = make_testyaml
    make_runs(run_name,
              yamlfile=yamlfile,
              cont_cvars=False)  # write run.txt files

    # code to test that run files are correct


def test_make_jobs(make_testyaml):
    run_name = 'test'
    yamlfile = make_testyaml
    make_jobs(run_name,
              yamlfile=yamlfile,
              cont_years=False,
              cont_cvars=False)  # write job.txt files

    # code to test that job files are correct


def test_make_subjobs(make_testyaml):
    run_name = 'test'
    yamlfile = make_testyaml
    make_subjobs(run_name,
                 yamlfile=yamlfile)  # write subjobs.sh

    # code to test that subjobs.sh is correct
