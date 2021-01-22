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
import glob
import numpy as np
import pandas as pd
import yaml
from numpy import genfromtxt

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


def make_dircts(dirct_project, run_name, dict_setup):
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
    # /init
    dirct_inits = os.path.join(dirct_project, 'inits', run_name)

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

        if not os.path.isdir(dirct_folder):
            os.mkdir(dirct_folder)

            for year in np.arange(years[0], years[1]+1):
                os.mkdir(os.path.join(dirct_folder, str(year)))

            cultivars = list()
            for var in np.arange(cvars):
                cultivar = 'var_' + str(var)
                cultivars.append(cultivar)

            for year in np.arange(years[0], years[1]+1):
                for var in cultivars:
                    os.mkdir(os.path.join(dirct_folder, str(year), str(var)))

        else:
            raise ValueError(f'directory {dirct_folder} already exists!')


def make_runs(init_yaml):
    """
    """
siteyears = pd.read_csv(dirct_project + 'weadata/siteyears_filtered.csv',
                        dtype={'site': str}, index_col=0)

custom_dict = {int(2): 'time',
               int(4): 'climate',
               int(8): 'management',
               int(12): 'init'}
standard_dict = {int(3): 'biology',
                 int(5): 'nitrogen',
                 int(6): 'solute',
                 int(7): 'soil',
                 int(9): 'drip',
                 int(10): 'water',
                 int(11): 'waterbound',
                 int(14): 'grid',
                 int(15): 'nod',
                 int(16): 'massbl'}

for i in np.arange(siteyears.shape[0]):
    site = siteyears.iloc[i, 0]
    year = str(siteyears.iloc[i, 1])

    # setting up directories
    init_dirct_wea = dirct_project + 'weadata/data/control/'
    init_dirct_stand = dirct_project + 'inits/standard_test/'
    init_dirct_custom = dirct_project + 'inits/custom/' + site + '_' + year + '/'

    custom_dict_loop = custom_dict.copy()
    standard_dict_loop = standard_dict.copy()
    for key, value in custom_dict_loop.items():
        custom_dict_loop[key] = init_dirct_custom + value + '.txt\n'
    for key, value in standard_dict_loop.items():
        standard_dict_loop[key] = init_dirct_stand + value + '.txt\n'

    # strings in run file
    cultivars = glob.glob(dirct_project + 'inits/var/*')

    for j in cultivars:
        var = j.split('/')[-1].split('.')[-2]
        output_dirct = full_dirct + year + '/' + var + '/'
        output_dict = {int(17): 'out1_' + site + '_' + year + '_' + var, 
                       int(18): 'out2_' + site + '_' + year + '_' + var,
                       int(19): 'out3',
                       int(20): 'out4',
                       int(21): 'out5',
                       int(22): 'out6',
                       int(23): 'massbl',
                       int(24): 'runoff'}

        for key, value in output_dict.items():
            output_dict[key] = output_dirct + value + '.txt\n'

        full_dict = {int(1): init_dirct_wea + site + '_' + year + '.txt\n',
                     int(13): j + '\n',  # loop through different var files
                     **custom_dict_loop,
                     **standard_dict_loop,
                     **output_dict}

        # combine strings
        keylist = sorted(full_dict.keys())
        strings = [full_dict[key] for key in keylist]

        # writing out run.txt file
        run = open(full_dirct + year + '/' + var + 
                   '/run_' + site + '_' + year + '_' + var + '.txt', 'w')
        run.writelines(strings)
        run.close()



def make_jobs(init_yaml):
    """
    """
cultivars = glob.glob(dirct_project + 'inits/var/*')
treatment = 'cont'

for year in np.arange(1961, 2006):
    for cvar in cultivars:
        var = cvar.split('/')[-1].split('.')[-2]
        logfile = str(year) + '_' + str(var) + '.log'
        str1 = '#!/bin/bash\n'
        str2 = '#PBS -l nodes=1:ppn=1\n'
        str3 = '#PBS -l walltime=08:00:00\n'
        str4 = '#PBS -m a\n'
        str5 = '#PBS -M ach315@uw.edu\n'
        str6 = '#PBS -N ' + treatment + '_' + str(i) + '_' + str(var) + '\n'
        str7 = '\n'
        str8 = 'FILES=' + full_dirct + str(year) + '/' + str(var) + '/*\n'
        str9 = 'cd ' + os.path.join(dict_runspecs['path_sims'],
                                    dict_runspecs['run_name'],
                                    str(year),
                                    str(cvar)) + '\n'
        str10 = 'touch ' + logfile + '\n'
        str11 = '\n'
        str12 = 'for file in $FILES\n'
        str13 = 'do\n'
        str14 = '\tfname=$(echo $file)\n'  # grab file name
        str15 = ('\tmaizsim_hash='
                 '$(git describe --dirty --always --tags)\n')  # githash
        str16 = f'\techo $fname,$maizsim_hash >> {logfile}\n'  # append to log
        str17 = '\tcd /home/disk/eos8/ach315/MAIZSIM\n'
        str18 = '\ttimeout 15m maizsim $file\n'
        str19 = 'done\n'

        strings = [str1, str2, str3, str4, str5, str6,
                   str7, str8, str9, str10, str11, str12,
                   str13, str14, str15, str16, str17, str18, str19]

        jobs = open(os.path.join(dict_runspecs['path_jobs'],
                                 str(year) + '_' + str(var) + '.job'), 'w')
        jobs.writelines(strings)
        jobs.close()


def make_subjobs(init_yaml):
    """
    """
    # step 3: create bash script that automates qsub jobs
    # TODO: still need to update this code
    dirct = os.path.join(dict_runspecs['path_jobs'], run)
                # TODO: this last run doesn't make sense
                # should be run name

    str1 = '#!/bin/bash\n'
    str2 = '\n'
    str3 = 'JOBS=' + dirct + '/*\n'
    str4 = '\n'
    str5 = 'for job in $JOBS\n'
    str6 = 'do\n'
    str7 = '\twhile [ `qstat | grep ach315 | wc -l` -ge 100 ]\n'
    str8 = '\tdo\n'
    str9 = '\tsleep 1\n'
    str10 = '\tdone\n'
    str11 = '\tqsub $job\n'
    str12 = 'done\n'

    strings = [str1, str2, str3, str4, str5, str6,
            str7, str8, str9, str10, str11, str12]

    subjobs = open(os.join(dirct, 'subjosb.sh', 'w')) # TODO: rename subjobs.sh as 
                                                    # subjobs_runame.sh
    subjobs.writelines(strings)
    subjobs.close()

    strings = [str1, str2, str3, str4, str5, str6, str7,
            str8, str9, str10, str11, str12, str13, str14]

    jobs = open(dirct + '/' + str(i) + '_' + str(var) + '.job', 'w')
    jobs.writelines(strings)
    jobs.close()

