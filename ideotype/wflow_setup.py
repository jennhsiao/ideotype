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


def read_inityaml(run_name, yamlfile=None):
    """
    Read in init_runame yaml file.

    yaml file inclues all setup info for a particular experiment run.

    Parameters
    ----------
    run_name: str
        Run name for particular batch of simulations.
    yamlfile: str
        default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.

    Returns
    -------
    dict_setup: dictionary
        Dictionary that only includes experiment setup info.

    """
    # Default situation
    if yamlfile is None:
        # Fetch yaml file with experiment setup specs
        # yaml files all stored in ideotype/data/inits/
        fname_init = os.path.join(DATA_PATH,
                                  'inits',
                                  'init_' + run_name + '.yml')
    # Manul input a test yamlfile to function for testing purposes
    else:
        fname_init = yamlfile

    if not os.path.isfile(fname_init):  # check whether init_.yml file exists
        raise ValueError(f'init param file {fname_init} does not exist!')

    # read in init param yaml file
    with open(fname_init, 'r') as pfile:
        dict_init = yaml.safe_load(pfile)

    dict_setup = dict_init['setup']
    dict_setup['params'] = dict_init['params']
    dict_setup['specs'] = dict_init['specs']

    return dict_setup


def make_dircts(run_name, yamlfile=None, cont_years=True, cont_cvars=True):
    """
    Make all required directories in experiment directory.

    Directories include experiment-specific subdirectories for:
    1. /inits
    2. /jobs
    3. /runs
    4. /sims

    Parameters
    ----------
    run_name: str
        Run name for specific batch of simualtions.
    yamlfile: str
        Default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.
    cont_years: Bool
        Default True
        How yaml file stores simulation years info.
        True: stored start and end year assuming all years in between.
        False: stores individual years (testing purposes)
    cont_cvars: Bool
        Default True
        How yaml file stores simulation cvars info.
        True: stored single number representing the total number of cultivas.
        Fals: stored specific cultivars (testing purposes).

    """
    # read in setup yaml file
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)

    # setup project directory
    dirct_project = dict_setup['path_project']

    # /inits
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

        # check if cultivar in yamlfile is continuous or not
        # False case mostly for testing purposes
        cultivars = list()
        if cont_cvars is True:
            cvars_iter = np.arange(cvars)
        else:
            cvars_iter = cvars

        # assemble cultivars
        for var in cvars_iter:
            cultivar = 'var_' + str(var)
            cultivars.append(cultivar)

        if not os.path.isdir(dirct_folder):
            # make /runs or /sims directory with run_name
            os.mkdir(dirct_folder)

            # check if years in yaml file are consecutive range or individual
            # False case mostly for testing
            if cont_years is True:
                years_iter = np.arange(years[0], years[1]+1)
            else:
                years_iter = years

            # create top level year directories
            for year in years_iter:
                os.mkdir(os.path.join(dirct_folder, str(year)))
                # create second layer cultivar directories
                for cultivar in cultivars:
                    os.mkdir(os.path.join(dirct_folder,
                                          str(year),
                                          str(cultivar)))

        else:
            raise ValueError(f'directory {dirct_folder} already exists!')


def make_runs(run_name, yamlfile=None, cont_cvars=True):
    """
    Create run.txt files in corresponding directories for experiment.

    Parameters
    ----------
    run_name: str
        run name for batch of experiments.
    yamlfile: str
        default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.
    cont_cvars: Bool
        Default True
        How yaml file stores simulation cvars info.
        True: stored single number representing the total number of cultivas.
        Fals: stored specific cultivars (testing purposes).

    """
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    dirct_project = dict_setup['path_project']

    dirct_runs = os.path.join(dirct_project, 'runs', run_name)

    # only execute if no run files already exist
    filelist = get_filelist(dirct_runs)
    if len(filelist) == 0:
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

        # check if cultivar in yamlfile is continuous or not
        # False case mostly for testing purposes
        cultivars = list()
        if cont_cvars is True:
            cvars_iter = np.arange(cvars)
        else:
            cvars_iter = cvars

        # assemble cultivars
        for var in cvars_iter:
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
                                            siteyear.split('_')[1],
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
        raise ValueError(f'run.txt files for run_name: "{run_name}"'
                         ' already exist!')


def make_jobs(run_name, yamlfile=None, cont_years=True, cont_cvars=True):
    """
    Create job.txt files in corresponding directories for experiment.

    Parameters
    ----------
    run_name: str
        run name for batch of experiments.
    yamlfile: str
        default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.
    cont_years: Bool
        Default True
        How yaml file stores simulation years info.
        True: stored start and end year assuming all years in between.
        False: stores individual years (testing purposes)
    cont_cvars: Bool
        Default True
        How yaml file stores simulation cvars info.
        True: stored single number representing the total number of cultivas.
        Fals: stored specific cultivars (testing purposes).

    """
    # read in setup yaml file
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    # setup project directory
    dirct_project = dict_setup['path_project']

    # point to jobs directory
    dirct_jobs = os.path.join(dirct_project, 'jobs', run_name)

    # only execute if no run files already exist
    filelist = get_filelist(dirct_jobs)
    if len(filelist) == 0:
        years = dict_setup['specs']['years']  # fetch from init_runame.yml
        cvars = dict_setup['specs']['cvars']  # fetch from init_runame.yml

        # check if cultivar in yamlfile is continuous or not
        # False case mostly for testing purposes
        cultivars = list()
        if cont_cvars is True:
            cvars_iter = np.arange(cvars)
        else:
            cvars_iter = cvars

        # assemble cultivars
        for var in cvars_iter:
            cultivar = 'var_' + str(var)
            cultivars.append(cultivar)

        # check if years in yaml file are consecutive range or individual
        # False case mostly for testing
        if cont_years is True:
            years_iter = np.arange(years[0], years[1]+1)
        else:
            years_iter = years

        # create a job script for each year_cultivar combination
        # for the job script to grab all run files of all valid sites
        # within that year
        for year in years_iter:
            for cvar in cultivars:
                logfile = str(year) + '_' + cvar + '.log'

                str1 = '#!/bin/bash\n'
                str2 = '#PBS -l nodes=1:ppn=1\n'
                str3 = '#PBS -l walltime=12:00:00\n'
                str4 = '#PBS -m a\n'
                str5 = '#PBS -M ach315@uw.edu\n'
                str6 = ('#PBS -N ' + run_name + '_' + str(year) +
                        '_' + cvar + '\n')
                str7 = '\n'
                str8 = 'FILES=' + os.path.join(dirct_project,
                                               'runs',
                                               run_name,
                                               str(year),
                                               cvar,
                                               '*') + '\n'
                str9 = '\n'
                str10 = 'cd ' + os.path.join(dirct_project,
                                             'sims',
                                             run_name,
                                             str(year),
                                             cvar) + '\n'
                str11 = 'touch ' + logfile + '\n'
                str12 = '\n'
                str13 = 'for file in $FILES\n'
                str14 = 'do\n'
                str15 = '\tfname=$(echo $file)\n'  # grab file name
                str16 = ('\tmaizsim_hash='  # grab git hash
                         '$(git describe --dirty --always --tags)\n')
                str17 = f'\techo $fname,$maizsim_hash >> {logfile}\n'  # append
                str18 = '\tcd /home/disk/eos8/ach315/MAIZSIM\n'
                str19 = '\ttimeout 15m maizsim $file\n'
                str20 = 'done\n'

                strings = [str1, str2, str3, str4, str5, str6,
                           str7, str8, str9, str10, str11, str12, str13,
                           str14, str15, str16, str17, str18, str19, str20]

                jobs = open(os.path.join(dirct_jobs,
                                         str(year) + '_' + cvar + '.job'), 'w')
                jobs.writelines(strings)
                jobs.close()

    else:
        raise ValueError(f'job.txt files for run_name: "{run_name}"'
                         ' already exist!')


def make_subjobs(run_name, yamlfile=None):
    """
    Create subjobs.sh bash script to runall corresponding jobs.

    Parameters
    ----------
    run_name: str
        run name for batch of experiments.
    yamlfile: str
        default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.

    """
    # read in setup yaml file
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    # setup project directory
    dirct_project = dict_setup['path_project']

    # point to jobs directory
    dirct_jobs = os.path.join(dirct_project, 'jobs')
    subjobs = 'subjobs_' + run_name + '.sh'

    if not os.path.exists(os.path.join(dirct_jobs, subjobs)):
        str1 = '#!/bin/bash\n'
        str2 = '\n'
        str3 = 'JOBS=' + os.path.join(dirct_jobs, run_name, '*') + '\n'
        str4 = '\n'
        str5 = 'for job in $JOBS\n'
        str6 = 'do\n'
        str7 = '\twhile [ `qstat | grep ach315 | wc -l` -ge 100 ]\n'
        str8 = '\tdo\n'
        str9 = '\t\tsleep 1\n'
        str10 = '\tdone\n'
        str11 = '\tqsub $job\n'
        str12 = 'done\n'

        strings = [str1, str2, str3, str4, str5, str6,
                   str7, str8, str9, str10, str11, str12]

        subjobs_script = open(os.path.join(dirct_jobs, subjobs), 'w')
        subjobs_script.writelines(strings)
        subjobs_script.close()

    else:
        raise ValueError(f'subjobs.sh for run_name: "{run_name}"'
                         ' already exists!')
