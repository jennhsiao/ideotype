import os
import glob
import argparse
import numpy as np
import pandas as pd

# TODO: add code to mkdir for custom init files here as well
# TODO: update some names to make code more readable
# TODO: test that output directories and run/job files are correct
# TODO: if directories already exist, print to terminal to show that
# TODO: update some hard-coded directories to point to yml file


# look into argparse
a = argparse.ArgumentParser(
    description=" help text here ___ "
)
a.add_argument(
    "folder",
    type=str,
    help="folder to create e.g. sims/ or runs/"
)

a.add_argument(
    "--dryrun",
    type=bool,
    default=False,
    help=("print info on directories that are going to be made"
          "rather than actually making them")
)

# get args
args = a.parse_args()


# step 0: setup directories
exp_dirct = '/home/disk/eos8/ach315/upscale/'  # experiment directory
folder = args.folder  # e.g. 'sims/' or 'runs/'
folder_name = 'opt_pdate/'

# check to see if folder exits, only execute if not
if not os.path.isdir(exp_dirct + folder):
    os.mkdir(exp_dirct + folder)
    full_dirct = exp_dirct + folder + folder_name
    os.mkdir(full_dirct)

    for year in np.arange(1961, 2006):
        os.mkdir(exp_dirct + folder + folder_name + str(year))

    cultivars = list()
    for var in np.arange(0, 100):
        cultivar = 'var_' + str(var)
        cultivars.append(cultivar)

    for year in np.arange(1961, 2006):
        for var in cultivars:
            os.mkdir(exp_dirct + folder + folder_name +
                     str(year) + '/' + str(var))

# step 1: create run.txt files
siteyears = pd.read_csv(exp_dirct + 'weadata/siteyears_filtered.csv',
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
    site = siteyears.iloc[i,0]
    year = str(siteyears.iloc[i,1])

    # setting up directories
    init_dirct_wea = exp_dirct + 'weadata/data/control/'
    init_dirct_stand = exp_dirct + 'inits/standard_test/'
    init_dirct_custom = exp_dirct + 'inits/custom/' + site + '_' + year + '/'

    custom_dict_loop = custom_dict.copy()
    standard_dict_loop = standard_dict.copy()
    for key, value in custom_dict_loop.items():
        custom_dict_loop[key] = init_dirct_custom + value + '.txt\n'
    for key, value in standard_dict_loop.items():
        standard_dict_loop[key] = init_dirct_stand + value + '.txt\n'

    # strings in run file
    cultivars = glob.glob(exp_dirct + 'inits/var/*')

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

        #print(strings)
        #assert False


 
        # writing out run.txt file
        run = open(full_dirct + year + '/' + var + 
                   '/run_' + site + '_' + year + '_' + var + '.txt', 'w')
        run.writelines(strings)
        run.close()


# step 2: create job files that execute batch of run files
dirct = exp_dirct + 'jobs/opt/'
cultivars = glob.glob(exp_dirct + 'inits/var/*')
treatment = 'cont'

for i in np.arange(1961,2006):
    for j in cultivars:
        var = j.split('/')[-1].split('.')[-2]
        str1 = '#!/bin/bash\n'
        str2 = '#PBS -l nodes=1:ppn=1\n'
        str3 = '#PBS -l walltime=08:00:00\n'
        str4 = '#PBS -m a\n'
        str5 = '#PBS -M ach315@uw.edu\n'
        str6 = '#PBS -N ' + treatment + '_' + str(i) + '_' + str(var) + '\n'
        str7 = '\n' 
        str8 = 'FILES=' + full_dirct + str(i)+ '/' + str(var) + '/*\n'
        str9 = '\n'
        str10 = 'for file in $FILES\n'
        str11 = 'do\n'
        str12 = '    cd /home/disk/eos8/ach315/MAIZSIM\n'
        str13 = '    timeout 15m maizsim $file\n'
        str14 = 'done\n'

        strings = [str1, str2, str3, str4, str5, str6, str7, str8, str9, str10, str11, str12, str13, str14]

        jobs = open(dirct + '/' + str(i) + '_' + str(var) + '.job', 'w')
        jobs.writelines(strings)
        jobs.close()

# step 3: create bash script that automates qsub jobs
# just wrote this directly thorugh terminal but probably want to code it up

