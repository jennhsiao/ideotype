"""Testing wflow setup."""

import os
import pytest
import yaml
from shutil import copyfile

from ideotype.utils import get_filelist
from ideotype.data import DATA_PATH
from ideotype.wflow_setup import (make_dircts, make_runs,
                                  make_jobs, make_subjobs)

# setup pointer to some default init files
dirct_default_init = '/home/disk/eos8/ach315/upscale/inits/'


@pytest.fixture(scope='module')
def make_testyaml(tmp_path_factory):
    """Update init_test.yml file on the fly with tmp_paths."""
    # link to init_test.yml
    tmp_path = tmp_path_factory.mktemp('ideotype_test')

    # create standard directory structure in the temp directory
    # the mirrors standard project directory
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

    # /init_standards
    dirct_default_standard = os.path.join(dirct_default_init,
                                          'standards',
                                          'opt')
    dirct_temp_standard = os.path.join(tmp_path,
                                       'inits',
                                       'standards')

    # list of standard init files to copy
    fname_standards = ['biology.txt',
                       'nitrogen.txt',
                       'drip.txt',
                       'water.txt',
                       'waterbound.txt',
                       'massbl.txt']

    # copy all standard init files to temp directory
    for fname in fname_standards:
        copyfile(
            os.path.join(dirct_default_standard, fname),
            os.path.join(dirct_temp_standard, fname))

    # /init_soils
    dirct_default_soil = os.path.join(dirct_default_init,
                                      'soils',
                                      'soils_1')
    dirct_temp_soil = os.path.join(tmp_path,
                                   'inits',
                                   'soils')

    # list of soil init files to copy
    fname_soils = ['grid.txt',
                   'nod.txt',
                   'soil.txt',
                   'solute.txt']

    # copy all standard init files to temp directory
    for fname in fname_soils:
        copyfile(
            os.path.join(dirct_default_soil, fname),
            os.path.join(dirct_temp_soil, fname))

    # create test_yaml file on the fly for testing purposes
    test_yaml = os.path.join(DATA_PATH, 'inits', 'init_test.yml')
    updated_yaml = os.path.join(tmp_path, 'init_test.yml')

    # check if init_test.yml exists
    if not os.path.exists(test_yaml):
        raise ValueError(f'{test_yaml} does not exist!')

    # read in init_test.yml
    with open(test_yaml, 'r') as pfile:
        dict_init = yaml.safe_load(pfile)

    # update certain paths with tmp_path
    dict_init['setup']['path_project'] = str(tmp_path)
    dict_init['setup']['path_init_standards'] = dirct_temp_standard
    dict_init['setup']['path_init_soils'] = dirct_temp_soil

    # overwrite existing init_test.yml file
    with open(updated_yaml, 'w') as outfile:
        yaml.dump(dict_init, outfile,
                  default_flow_style=False, sort_keys=False)

    return updated_yaml


def test_make_dircts(make_testyaml):
    """Make test directories under temporary path."""
    run_name = 'test'
    yamlfile = make_testyaml
    make_dircts(run_name,
                yamlfile=yamlfile,
                cont_years=False,
                cont_cvars=False)  # make test directories

    # code to test directories are correct
    #get_filelist()


def test_make_runs(make_testyaml):
    """Make test run.txt within temporary directories."""
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
