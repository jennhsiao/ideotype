"""
Setup directories & files for new experiment.

Based on init_runame.yml file, sets up:
- inits/
- runs/
- jobs/
- sims/
- subjob.s

"""
import os
import numpy as np
import yaml
from numpy import genfromtxt
from ideotype.utils import get_filelist

from ideotype.data import DATA_PATH


def read_yaml(run_name):
    """
    Read in init_runame yaml file.

    yaml file inclues all setup info for a particular experiment run.

    Parameters
    ----------
    run_name: str
        Run name for particular batch of simulations.

    Returns
    -------
    dict_setup: dictionary
        Dictionary that only includes experiment setup info.

    """
    # Fetch yaml file with experiment setup specs
    # yaml files all stored in ideotype/data/inits/
    fname_param = os.path.join(DATA_PATH, 'inits', 'init_' + run_name + '.yml')

    if not os.path.isfile(fname_param):  # check whether init_.yml file exists
        raise ValueError(f'init param file {fname_param} does not exist')

    # read in init param yaml file
    with open(fname_param, 'r') as pfile:
        dict_init = yaml.safe_load(pfile)

    dict_setup = dict_init['setup']
    dict_setup['params'] = dict_init['params']
    dict_setup['specs'] = dict_init['specs']

    return dict_setup


def make_dircts(run_name, dict_setup):
    """
    Make all required directories in experiment directory.

    Directories include experiment-specific subdirectories for:
    1. /inits
    2. /jobs
    3. /runs
    4. /sims

    Parameters
    ----------
    dirct_project: str
        Root directory for project.
    run_name: str
        Run name for specific batch of simualtions.
    dict_setup: dict
        Dictionary that includes setup info.

    """
    # setup project directory
    dirct_project = dict_setup['path_project']

    # /init
    dirct_inits = os.path.join(dirct_project, 'inits', 'customs', run_name)

    data = genfromtxt(dict_setup['siteyears'],
                      delimiter=',',
                      skip_header=1,
                      usecols=(0, 1),
                      dtype=('U6', int, int, 'U10'))

    siteyears = []
    for row in data:
        siteyears.append(str(row[0]) + '_' + str(row[1]))

    # Check if folder exits, only execute if not
    if not os.path.isdir(dirct_inits):
        os.mkdir(dirct_inits)
        for siteyear in siteyears:
            os.mkdir(os.path.join(dirct_inits, siteyear))

    else:
        raise ValueError(f'directory {dirct_inits} already exists!')

    # /jobs
    dirct_jobs = os.path.join(dirct_project, 'jobs', run_name)

    if not os.path.isdir(dirct_jobs):
        os.mkdir(dirct_jobs)

    else:
        raise ValueError(f'directory {dirct_jobs} already exists!')

    # /runs & /sims
    for folder in (['runs', 'sims']):
        dirct_folder = os.path.join(dirct_project, folder, run_name)
        years = dict_setup['specs']['years']  # fetch from init_runame.yml
        cvars = dict_setup['specs']['cvars']  # fetch from init_runame.yml

        # assemble cultivars
        cultivars = list()
        for var in np.arange(cvars):
            cultivar = 'var_' + str(var)
            cultivars.append(cultivar)

        if not os.path.isdir(dirct_folder):
            os.mkdir(dirct_folder)

            for year in np.arange(years[0], years[1]+1):
                os.mkdir(os.path.join(dirct_folder, str(year)))

                for cultivar in cultivars:
                    os.mkdir(os.path.join(dirct_folder,
                                          str(year),
                                          str(cultivar)))

        else:
            raise ValueError(f'directory {dirct_folder} already exists!')


def make_runs(run_name, dict_setup):
    """
    Create run.txt files in corresponding directories for experiment.

    Parameters
    ----------
    init_yaml: str
        init_runame.yml file that includes experiment setup and run specs.
    dict_setup: dict
        Dictionary with setup info & run specs.
        Use read_yaml function to get this dictionary.

    """
    # TODO: should check if these run files exist already in case you overwrite
    # setup project directory
    dirct_project = dict_setup['path_project']

    dirct_runs = os.path.join(dirct_project, 'runs', run_name)
    filelist = get_filelist(dirct_runs)
    if len(filelist=0):



        # read in dict_setup to fetch site-years info
        data = genfromtxt(dict_setup['siteyears'],
                        delimiter=',',
                        skip_header=1,
                        usecols=(0, 1),
                        dtype=('U6', int, int, 'U10'))

        # setup site_years
        siteyears = []
        for row in data:
            siteyears.append(str(row[0]) + '_' + str(row[1]))

        # setup cultivars
        cvars = dict_setup['specs']['cvars']  # fetch cultivar numbers
        cultivars = list()
        for var in np.arange(cvars):
            cultivar = 'var_' + str(var)
            cultivars.append(cultivar)

        # setup up directories
        dirct_init_wea = dict_setup['path_wea']
        dirct_init_stand = dict_setup['path_init_standards']
        dirct_init_soils = dict_setup['path_init_soils']

        dict_standard = {int(3): 'biology',
                        int(5): 'nitrogen',
                        int(9): 'drip',
                        int(10): 'water',
                        int(11): 'waterbound',
                        int(16): 'massbl'}
        dict_soils = {int(14): 'grid',
                    int(15): 'nod',
                    int(7): 'soil',
                    int(6): 'solute'}
        dict_custom = {int(2): 'time',
                    int(4): 'climate',
                    int(8): 'management',
                    int(12): 'init'}

        # itemize dictionary items into paths
        dict_standard_loop = dict_standard.copy()
        dict_soils_loop = dict_soils.copy()
        for key, value in dict_standard_loop.items():
            dict_standard_loop[key] = os.path.join(
                dirct_init_stand, f'{value}.txt') + '\n'
        for key, value in dict_soils_loop.items():
            dict_soils_loop[key] = os.path.join(
                dirct_init_soils, f'{value}.txt') + '\n'

        # loop through siteyears
        for siteyear in siteyears:
            # setup siteyear-specific directory
            dirct_init_custom = os.path.join(dirct_project,
                                            'inits',
                                            'customs',
                                            run_name,
                                            siteyear)

            # itemize dictionary items into paths
            dict_custom_loop = dict_custom.copy()
            for key, value in dict_custom_loop.items():
                dict_custom_loop[key] = os.path.join(
                    dirct_init_custom, f'{value}.txt') + '\n'

            # loop through cultivars
            for cultivar in cultivars:
                dirct_output = os.path.join(dirct_project,
                                            'sims',
                                            run_name,
                                            siteyear.split('_')[1],  # parse year
                                            cultivar)
                dict_output = {int(17): 'out1_' + siteyear + '_' + cultivar,
                            int(18): 'out2_' + siteyear + '_' + cultivar,
                            int(19): 'out3',
                            int(20): 'out4',
                            int(21): 'out5',
                            int(22): 'out6',
                            int(23): 'massbl',
                            int(24): 'runoff'}

                for key, value in dict_output.items():
                    dict_output[key] = os.path.join(dirct_output,
                                                    f'{value}.txt') + '\n'

                dict_all = {int(1): os.path.join(dirct_init_wea,
                                                f'{siteyear}.txt') + '\n',
                            int(13): os.path.join(dirct_project,
                                                'inits',
                                                'cultivars',
                                                run_name,
                                                cultivar) + '\n',
                            **dict_standard_loop,
                            **dict_soils_loop,
                            **dict_custom_loop,
                            **dict_output}

                # combine strings
                keylist = sorted(dict_all.keys())
                strings = [dict_all[key] for key in keylist]

                # writing out run.txt file
                run = open(os.path.join(
                    dirct_project,
                    'runs',
                    run_name,
                    siteyear.split('_')[1],  # parse year
                    cultivar,
                    'run_' + siteyear + '_' + cultivar + '.txt'), 'w')
                run.writelines(strings)
                run.close()
    else:
        raise ValueError(f'run.txt files for {run_name} already exists!')


def make_jobs(init_yaml):
    """
    """
    pass


def make_subjobs(init_yaml):
    """
    """
    pass
