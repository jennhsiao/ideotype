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

from ideotype.data import DATA_PATH


def read_yaml():
    """
    """
    # STEP 0: fetch yaml file with experiment setup specs
    fname_param = os.path.join(DATA_PATH, 'inits', 'init_' + run_name + '.yml')

    if not os.path.isfile(fname_param):  # check whether init_.yml file exists
        raise ValueError(f'init param file {fname_param} does not exist')

    with open(fname_param, 'r') as pfile:  # read in init param yaml file
        dict_param = yaml.safe_load(pfile)

    dict_runspecs = dict_param['setup']


def make_dircts():
    """
    """
    # STEP 0: setup directories
    # TODO: add path_exp to yaml file and fetch dirct_exp directly
    dirct_exp = '/home/disk/eos8/ach315/upscale/'  # experiment directory
    folder = args.folder  # e.g. 'sims/' or 'runs/'
    folder_name = 'opt_pdate/'

    # check to see if folder exits, only execute if not
    if not os.path.isdir(dirct_exp + folder):
        os.mkdir(dirct_exp + folder)
        full_dirct = dirct_exp + folder + folder_name
        os.mkdir(full_dirct)

        for year in np.arange(1961, 2006):
            os.mkdir(dirct_exp + folder + folder_name + str(year))

        cultivars = list()
        for var in np.arange(0, 100):
            cultivar = 'var_' + str(var)
            cultivars.append(cultivar)

        for year in np.arange(1961, 2006):
            for var in cultivars:
                os.mkdir(dirct_exp + folder + folder_name +
                        str(year) + '/' + str(var))
    else:
        raise ValueError(f'directory {dirct_exp + folder} already exists!')


def make_runs(init_yaml):
    """
    """
siteyears = pd.read_csv(dirct_exp + 'weadata/siteyears_filtered.csv',
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
    init_dirct_wea = dirct_exp + 'weadata/data/control/'
    init_dirct_stand = dirct_exp + 'inits/standard_test/'
    init_dirct_custom = dirct_exp + 'inits/custom/' + site + '_' + year + '/'

    custom_dict_loop = custom_dict.copy()
    standard_dict_loop = standard_dict.copy()
    for key, value in custom_dict_loop.items():
        custom_dict_loop[key] = init_dirct_custom + value + '.txt\n'
    for key, value in standard_dict_loop.items():
        standard_dict_loop[key] = init_dirct_stand + value + '.txt\n'

    # strings in run file
    cultivars = glob.glob(dirct_exp + 'inits/var/*')

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
cultivars = glob.glob(dirct_exp + 'inits/var/*')
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

