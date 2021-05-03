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
import pandas as pd
import yaml
from numpy import genfromtxt

from ideotype.utils import (get_filelist,
                            stomata_waterstress,
                            estimate_pdate)
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

    if not dict_init['setup']['run_name'] == run_name:
        raise ValueError('mismatch run_name between yaml file name'
                         'and setup record within yaml file!')

    dict_setup = dict_init['setup']
    dict_setup['params'] = dict_init['params']
    dict_setup['specs'] = dict_init['specs']
    dict_setup['init'] = dict_init['init']
    dict_setup['time'] = dict_init['time']
    dict_setup['climate'] = dict_init['climate']
    dict_setup['management'] = dict_init['management']
    dict_setup['cultivar'] = dict_init['cultivar']

    return dict_setup


def read_siteinfo(file_siteinfo, file_siteyears):
    """
    Read in site info and siteyears.

    Parameters:
    -----------
    file_siteinfo : str
        file path for site info
    file_siteyears : str
        file path for siteyears info

    Returns:
    --------
    site_info : pd dataframe
    siteyears : pd dataframe

    """
    site_info = pd.read_csv(file_siteinfo,
                            dtype={'USAF': str},
                            usecols=[0, 1, 3, 4, 8, 9, 10])
    site_info.columns = ['site', 'class', 'station',
                         'state', 'tzone', 'lat', 'lon']
    siteyears = pd.read_csv(file_siteyears, dtype=str)

    return site_info, siteyears


def make_dircts(run_name, yamlfile=None, cont_years=True, cont_cvars=True):
    """
    Make all required directories in experiment directory.

    Directories include experiment-specific subdirectories for:
    1. /inits
        1.1 /inits/customs
        1.2 /inits/cultivars
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

    # /inits - cultivars
    dirct_inits_cultivars = os.path.join(dirct_project,
                                         'inits',
                                         'cultivars',
                                         run_name)

    # Check if folder exits, only execute if not
    if not os.path.isdir(dirct_inits_cultivars):
        os.mkdir(dirct_inits_cultivars)
    else:
        raise ValueError(f'directory {dirct_inits_cultivars} already exists!')

    # /jobs
    dirct_jobs = os.path.join(dirct_project, 'jobs', run_name)

    if not os.path.isdir(dirct_jobs):
        os.mkdir(dirct_jobs)
    else:
        raise ValueError(f'directory {dirct_jobs} already exists!')

    # /inits/customs, /runs & /sims
    for folder in (['inits', 'customs'], ['runs'], ['sims']):
        dirct_folder = os.path.join(dirct_project, *folder, run_name)
        years = dict_setup['specs']['years']  # fetch from init_runame.yml
        cvars = dict_setup['specs']['cvars']  # fetch from init_runame.yml

        # check if cultivar in yamlfile is continuous or not
        # False case mostly for testing purposes
        cultivars = list()
        if cont_cvars is True:
            cvars_iter = np.arange(cvars[0])
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


def make_inits(run_name, yamlfile=None, cont_cvars=True):
    """
    Create custom init files for MAIZSIM sims.

    Parameters
    ----------
    run_name: str
        run name for batch of experiments.
    yamlfile: str
        default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.

    Returns
    -------
    init.txt
    time.txt
    climate.txt
    management.txt

    """
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    dirct_project = dict_setup['path_project']
    dirct_init = os.path.join(dirct_project, 'inits', 'customs', run_name)

    # only execute if no run files already exist
    filelist = get_filelist(os.path.expanduser(dirct_init))

    if len(filelist) == 0:
        # read in site_info & siteyears info
        if dict_setup['base_path'] == 'testing':
            basepath = DATA_PATH
            fpath_siteinfo = os.path.join(basepath,
                                          *dict_setup['site_info'])
            fpath_siteyears = os.path.join(basepath,
                                           *dict_setup['siteyears'])
            fpath_params = os.path.join(basepath,
                                        *dict_setup['path_params'])
            fpath_weas = os.path.join(basepath,
                                      *dict_setup['path_wea'])

        else:
            fpath_siteinfo = os.path.join(dict_setup['path_project'],
                                          *dict_setup['site_info'])
            fpath_siteyears = os.path.join(dict_setup['path_project'],
                                           *dict_setup['siteyears'])
            fpath_params = os.path.join(dict_setup['path_project'],
                                        *dict_setup['path_params'])
            fpath_weas = os.path.join(dict_setup['path_project'],
                                      *dict_setup['path_wea'])

        df_siteinfo, df_siteyears = read_siteinfo(fpath_siteinfo,
                                                  fpath_siteyears)
        df_params = pd.read_csv(fpath_params)

        # fetch & setup site-years info
        data = genfromtxt(fpath_siteyears,
                          delimiter=',',
                          skip_header=1,
                          usecols=(0, 1),
                          dtype=('U6', int, int, 'U10'))
        siteyears = []
        for row in data:
            siteyears.append(str(row[0]) + '_' + str(row[1]))

        # setup cultivars
        cvars = dict_setup['specs']['cvars']  # fetch cultivar numbers

        # check if cultivar in yamlfile is continuous or not
        # False case for control sim & testing purposes
        cultivars = list()
        if cont_cvars is True:
            cvars_iter = np.arange(cvars[0])
        else:
            cvars_iter = cvars

        # assemble cultivars
        for var in cvars_iter:
            cultivar = 'var_' + str(var)
            cultivars.append(cultivar)

        # loop through site-years
        for siteyear in siteyears:
            site = siteyear.split('_')[0]
            year = siteyear.split('_')[1]
            lat = df_siteinfo[df_siteinfo.site == site].lat.item()
            lon = df_siteinfo[df_siteinfo.site == site].lon.item()

            # loop through cultivars
            for cultivar in cultivars:
                # *** init.txt
                init_txt = open(os.path.join(dirct_init,
                                             year,
                                             cultivar,
                                             'init.txt'), 'w')

                # Dynamic pdate but gdd not perturbed
                if dict_setup['init']['plant_date'] == 'dynamic':
                    # use default gdd_threhold value
                    date_start, date_plant = estimate_pdate(fpath_weas,
                                                            site,
                                                            year)
                # Dynamic pdate with perturbed gdd
                elif dict_setup['init']['plant_date'] == 'dynamic_perturbed':
                    # use perturbed gdd value
                    gdd = df_params.loc[int(cultivar.split('_')[-1]), 'gdd']
                    date_start, date_plant = estimate_pdate(fpath_weas,
                                                            site,
                                                            year,
                                                            gdd_threshold=gdd)
                # Standard pdate across asll sites
                else:
                    sowing = f'{dict_setup["init"]["plant_date"]}{year}'
                    sowing = "'" + sowing + "'"  # requires single quote
                    start = f'{dict_setup["init"]["start_date"]}{year}'
                    start = "'" + start + "'"

                end = f'{dict_setup["init"]["end_date"]}{year}'
                end = "'" + end + "'"

                # set up init.txt text strings
                pop = df_params.loc[int(cultivar.split('_')[-1]), 'pop']
                str1 = '*** initialization data ***\n'
                str2 = ('poprow\trowsp\tplant_density\trowang\t'
                        'x_seed\ty_seed\ttab\tCEC\teomult\tco2\n')
                str3 = (f'{pop*(dict_setup["init"]["rowsp"])/100:.1f}\t'
                        f'{dict_setup["init"]["rowsp"]:.1f}\t'
                        f'{pop:.1f}\t'
                        f'{dict_setup["init"]["rowang"]:.1f}\t'
                        f'{dict_setup["init"]["x_seed"]:.1f}\t'
                        f'{dict_setup["init"]["y_seed"]:.1f}\t'
                        f'{dict_setup["init"]["cec"]:.2f}\t'
                        f'{dict_setup["init"]["eomult"]:.2f}\t'
                        f'{dict_setup["init"]["co2"]:.0f}\n')
                str4 = 'latitude\tlongitude\taltitude\n'
                str5 = (f'{lat:.2f}\t'
                        f'{lon:.2f}\t'
                        f'{dict_setup["init"]["alt"]:.2f}\n')
                str6 = 'autoirrigate\n'
                str7 = f'{dict_setup["init"]["irrigate"]}\n'
                str8 = 'begin\tsowing\tend\ttimestep (mins)\n'
                str9 = (start + '\t' + sowing + '\t' + end + '\t'
                        f'{dict_setup["init"]["timestep"]:.0f}\n')
                str10 = ('output soils data (g03, g04, g05, and g06 files)'
                         ' 1 if true\n')
                str11 = 'no soil files\toutputsoil files\n'
                if dict_setup["init"]["soil"]:
                    str12 = '0\t1\n'
                else:
                    str12 = '1\t0\n'

                strings = [str1, str2, str3, str4, str5, str6,
                           str7, str8, str9, str10, str11, str12]
                init_txt.writelines(strings)
                init_txt.close()

                # *** time.txt
                time_txt = open(os.path.join(dirct_init,
                                             year,
                                             cultivar,
                                             'time.txt'), 'w')

                # set up text strings
                str1 = '*** synchronizer information ***\n'
                str2 = 'initial time\tdt\tdtmin\tdmul1\tdmul2\ttfin\n'
                str3 = (start + '\t' +
                        f'{dict_setup["time"]["dt"]}\t'
                        f'{dict_setup["time"]["dt_min"]}\t'
                        f'{dict_setup["time"]["dmul1"]}\t'
                        f'{dict_setup["time"]["dmul2"]}\t' +
                        end + '\n')
                str4 = 'output variables, 1 if true\tDaily\tHourly\n'
                if dict_setup['time']['output_timestep'] == 'hourly':
                    output_timestep = '0\t1\n'
                else:
                    output_timestep = '1\t0\n'
                str5 = output_timestep
                str6 = 'weather data, 1 if true\tDaily\tHourly\n'
                if dict_setup['time']['input_timestep'] == 'hourly':
                    input_timestep = '0\t1\n'
                else:
                    input_timestep = '1\t0\n'
                str7 = input_timestep

                strings = [str1, str2, str3, str4, str5, str6, str7]
                time_txt.writelines(strings)
                time_txt.close()

                # *** climate.txt
                climate_txt = open(os.path.join(dirct_init,
                                                year,
                                                cultivar,
                                                'climate.txt'), 'w')

                # put together txt strings
                str1 = '*** standard meteorological data ***\n'
                str2 = 'latitude\n'
                str3 = f'{lat}\n'
                str4 = ('daily bulb temp, daily wind, rain intensity, '
                        'daily conc, furrow, relative humidity, co2\n')
                str5 = (f'{dict_setup["climate"]["daily_bulb"]}\t'
                        f'{dict_setup["climate"]["daily_wind"]}\t'
                        f'{dict_setup["climate"]["rain_intensity"]}\t'
                        f'{dict_setup["climate"]["daily_conc"]}\t'
                        f'{dict_setup["climate"]["furrow"]}\t'
                        f'{dict_setup["climate"]["relative_humidity"]}\t'
                        f'{dict_setup["climate"]["daily_co2"]}\n')
                str6 = ('parameters for unit conversion:'
                        'BSOLAR BTEMP ATEMP ERAIN BWIND BIR\n')
                str7 = 'BSOLAR is 1e6/3600 to go from jm-2h-1 to wm-2\n'
                str8 = (f'{dict_setup["climate"]["bsolar"]}\t'
                        f'{dict_setup["climate"]["btemp"]}\t'
                        f'{dict_setup["climate"]["atemp"]}\t'
                        f'{dict_setup["climate"]["erain"]}\t'
                        f'{dict_setup["climate"]["bwind"]}\t'
                        f'{dict_setup["climate"]["bir"]}\n')
                str9 = 'average values for the site\n'
                str10 = 'WINDA\tIRAV\tConc\tCO2\n'
                str11 = (f'{dict_setup["climate"]["winda"]}\t'
                         f'{dict_setup["climate"]["conc"]}\n')

                strings = [str1, str2, str3, str4, str5,
                           str6, str7, str8, str9, str10, str11]
                climate_txt.writelines(strings)
                climate_txt.close()

                # *** management.txt
                management_txt = open(os.path.join(dirct_init,
                                                   year,
                                                   cultivar,
                                                   'management.txt'), 'w')

                # addressing N application date according to dynamic pdate
                sowing_date = pd.to_datetime(sowing, format="'%m/%d/%Y'")
                appl_date1 = sowing_date + pd.DateOffset(days=14)
                appl_date2 = sowing_date + pd.DateOffset(days=14+30)
                appl_time1 = appl_date1.strftime("'%m/%d/%Y'")
                appl_time2 = appl_date2.strftime("'%m/%d/%Y'")

                # put together txt strings
                str1 = '*** script for chemical application module ***\n'
                str2 = 'number of fertilizer applicaitons (max=25)\n'
                str3 = f'{dict_setup["management"]["appl_num"]}\n'
                str4 = ('appl_time(i)\tappl_mg(i)\tappl_depth(cm)\t'
                        'residue_C\tresidue_N\n')
                str5 = (f'{appl_time1}\t'
                        f'{dict_setup["management"]["appl_mg"]}\t'
                        f'{dict_setup["management"]["appl_depth"]}\t'
                        f'{dict_setup["management"]["residue_C"]}\t'
                        f'{dict_setup["management"]["residue_N"]}\n')
                str6 = (f'{appl_time2}\t'
                        f'{dict_setup["management"]["appl_mg"]}\t'
                        f'{dict_setup["management"]["appl_depth"]}\t'
                        f'{dict_setup["management"]["residue_C"]}\t'
                        f'{dict_setup["management"]["residue_N"]}\n')

                strings = [str1, str2, str3, str4, str5, str6]
                management_txt.writelines(strings)
                management_txt.close()

    else:
        raise ValueError(f'custom initial files for run_name: "{run_name}"'
                         ' already exist!')


def make_cultivars(run_name, yamlfile=None, cont_cvars=True):
    """
    Create cultivar files based on perturbed combinations of parameters.

    Parameters
    ----------
    run_name : str
        run name for batch of experiments.
    yamlfile: str
        default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.
    cont_cvars : Bool
        Default True.
        How yaml file stores simulation cvars info.
        True: stored single number representing the total number of cultivas.
        False: stored specific cultivars (testing purposes).

    Returns
    -------
    var.txt

    """
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    dirct_project = dict_setup['path_project']
    dirct_cvars = os.path.join(dirct_project, 'inits', 'cultivars', run_name)

    # set up linear mod to estimate sf from phyf
    mod_intercept, mod_coef = stomata_waterstress()

    if dict_setup['base_path'] == 'testing':
        fpath_params = os.path.join(DATA_PATH, *dict_setup['path_params'])
    else:
        fpath_params = os.path.join(dict_setup['path_project'],
                                    *dict_setup['path_params'])

    df_params = pd.read_csv(fpath_params)

    filelist = get_filelist(os.path.expanduser(dirct_cvars))
    if len(filelist) == 0:
        if cont_cvars is True:
            cvars = np.arange(dict_setup['specs']['cvars'][0])
        else:
            cvars = dict_setup['specs']['cvars']

        for cvar in cvars:
            [
                g1, vcmax, jmax, phyf,  # physiology
                staygreen, juv_leaves, rmax_ltar,  # phenology
                lm_min, laf,  # morphology
                gdd, pop  # management
            ] = df_params.iloc[cvar, :]

            cvar_txt = open(os.path.join(dirct_cvars,
                            'var_' + str(cvar) + '.txt'), 'w')

            str1 = '*** Corn growth simulation for US maize simualtion ***\n'
            str2 = f'cultivar: {cvar}\n'
            str3 = ('juv_leaf\tdaylen_sen\tstaygreen\t'
                    'LM_min\tRmax_LTAR\tRmax_LTIR\tphyllo\n')
            str4 = '\n'
            str5 = (f'{juv_leaves:.0f}\t'
                    f'{dict_setup["cultivar"]["daylen_sen"]:.0f}\t'
                    f'{staygreen:.2f}\t'
                    f'{lm_min:.0f}\t'
                    f'{rmax_ltar:.2f}\t'
                    f'{rmax_ltar*2:.2f}\t'
                    f'{dict_setup["cultivar"]["phyllo"]:.0f}\n')
            str6 = '[SoilRoot]\n'
            str7 = '*** water uptake parameter information ***\n'
            str8 = 'RRRM\tRRRY\tRVRL\n'
            str9 = (f'{dict_setup["cultivar"]["rrrm"]:.2f}\t'
                    f'{dict_setup["cultivar"]["rrry"]:.2f}\t'
                    f'{dict_setup["cultivar"]["rvrl"]:.2f}\n')
            str10 = 'ALPM\tALPY\tRTWL\tRtMinWtPerUnitArea\n'
            str11 = (f'{dict_setup["cultivar"]["alpm"]:.2f}\t'
                     f'{dict_setup["cultivar"]["alpy"]:.2f}\t'
                     f'{dict_setup["cultivar"]["rtwl"]:.7f}\t'
                     f'{dict_setup["cultivar"]["rtminwtperunitarea"]:.4f}\n')
            str12 = '[RootDiff]\n'
            str13 = '*** root mover parameter information ***\n'
            str14 = 'EPSI\tlUpW\tCourMax\n'
            str15 = (f'{dict_setup["cultivar"]["epsi"]:.0f}\t'
                     f'{dict_setup["cultivar"]["lupw"]:.0f}\t'
                     f'{dict_setup["cultivar"]["courmax"]:.0f}\n')
            str16 = 'Diffusivity and geotrophic velocity\n'
            str17 = (f'{dict_setup["cultivar"]["diffgeo1"]:.1f}\t'
                     f'{dict_setup["cultivar"]["diffgeo2"]:.1f}\t'
                     f'{dict_setup["cultivar"]["diffgeo3"]:.1f}\n')
            str18 = '[SoilNitrogen]\n'
            str19 = '*** nitrogen root uptake parameter infromation ***\n'
            str20 = 'ISINK\tRroot\n'
            str21 = (f'{dict_setup["cultivar"]["isink"]:.0f}\t'
                     f'{dict_setup["cultivar"]["rroot"]:.2f}\n')
            str22 = 'ConstI\tConstk\tCmin0\n'
            str23 = (f'{dict_setup["cultivar"]["consti_1"]:.2f}\t'
                     f'{dict_setup["cultivar"]["constk_1"]:.2f}\t'
                     f'{dict_setup["cultivar"]["cmin0_1"]:.2f}\n')
            str24 = (f'{dict_setup["cultivar"]["consti_2"]:.2f}\t'
                     f'{dict_setup["cultivar"]["constk_2"]:.2f}\t'
                     f'{dict_setup["cultivar"]["cmin0_2"]:.2f}\n')
            str25 = '[Gas_Exchange Species Parameters]\n'
            str26 = '*** for photosynthesis calculations ***\n'
            str27 = ('EaVP\tEaVc\tEaj\tHj\tSj\t'
                     'Vpm25\tVcm25\tJm25\tRd25\tEar\tg0\tg1\n')
            str28 = (f'{dict_setup["cultivar"]["eavp"]:.0f}\t'
                     f'{dict_setup["cultivar"]["eavc"]:.0f}\t'
                     f'{dict_setup["cultivar"]["eaj"]:.0f}\t'
                     f'{dict_setup["cultivar"]["hj"]:.0f}\t'
                     f'{dict_setup["cultivar"]["sj"]:.0f}\t'
                     f'{dict_setup["cultivar"]["vpm_25"]:.0f}\t'
                     f'{vcmax:.0f}\t'
                     f'{jmax:.0f}\t'
                     f'{dict_setup["cultivar"]["rd_25"]:.0f}\t'
                     f'{dict_setup["cultivar"]["ear"]:.0f}\t'
                     f'{dict_setup["cultivar"]["g0"]:.2f}\t'
                     f'{g1:.2f}\n')
            str29 = '*** second set of parameters for photosynthesis ***\n'
            str30 = 'f\tscatt\tKc_25\tKo_25\tKp_25\tgbs\tgi\tgamma1\n'
            str31 = (f'{dict_setup["cultivar"]["f"]:.2f}\t'
                     f'{dict_setup["cultivar"]["scatt"]:.2f}\t'
                     f'{dict_setup["cultivar"]["Kc_25"]:.0f}\t'
                     f'{dict_setup["cultivar"]["Ko_25"]:.0f}\t'
                     f'{dict_setup["cultivar"]["Kp_25"]:.0f}\t'
                     f'{dict_setup["cultivar"]["gbs"]:.3f}\t'
                     f'{dict_setup["cultivar"]["gi"]:.2f}\t'
                     f'{dict_setup["cultivar"]["gamma1"]:.2f}\n')
            str32 = '*** third set of photosynthesis parameters ***\n'
            str33 = ('gamma_gsw\tsensitivity (sf)\t'
                     'ref_potential (phyla, bars)\t'
                     'stoma_ratio\twidfct\tleaf_wid (m)\n')
            str34 = (f'{dict_setup["cultivar"]["gamma_gsw"]:.1f}\t'
                     f'{mod_intercept+phyf*mod_coef:.1f}\t'
                     f'{phyf:.1f}\t'
                     f'{dict_setup["cultivar"]["stomata_ratio"]:.1f}\t'
                     f'{dict_setup["cultivar"]["widfct"]:.2f}\t'
                     f'{dict_setup["cultivar"]["leaf_wid"]:.2f}\n')
            str35 = ('**** seconday parameters for '
                     'miscelanioius equations ****\n')
            str36 = 'Ci/Ca\tSC_param\tBLC_param\n'
            str37 = (f'{dict_setup["cultivar"]["cica_ratio"]:.1f}\t'
                     f'{dict_setup["cultivar"]["SC_param"]:.2f}\t'
                     f'{dict_setup["cultivar"]["BLC_param"]:.1f}\n')
            str38 = '*** Q10 params for respiration and leaf senescence ***\n'
            str39 = 'Q10MR\tQ10LeafSenescence\n'
            str40 = (f'{dict_setup["cultivar"]["Q10MR"]:.1f}\t'
                     f'{dict_setup["cultivar"]["Q10Senescence"]:.1f}\n')
            str41 = '*** Leaf morphology factors ***\n'
            str42 = 'LAF\tWLRATIO\tA_LW\n'
            str43 = (f'{laf:.2f}\t'
                     f'{dict_setup["cultivar"]["WLRATIO"]:.3f}\t'
                     f'{dict_setup["cultivar"]["A_LW"]:.2f}\n')
            str44 = '*** temperature factors for growth ***\n'
            str45 = 'T_base\tT_opt\tT_ceil\tT_opt_GDD\n'
            str46 = (f'{dict_setup["cultivar"]["T_base"]:.1f}\t'
                     f'{dict_setup["cultivar"]["T_opt"]:.1f}\t'
                     f'{dict_setup["cultivar"]["T_ceil"]:.1f}\t'
                     f'{dict_setup["cultivar"]["T_opt_GDD"]:.1f}\n')

            strings = [str1, str2, str3, str4, str5, str6, str7, str8,
                       str9, str10, str11, str12, str13, str14, str15, str16,
                       str17, str18, str19, str20, str21, str22, str23, str24,
                       str25, str26, str27, str28, str29, str30, str31, str32,
                       str33, str34, str35, str36, str37, str38, str39, str40,
                       str41, str42, str43, str44, str45, str46]
            cvar_txt.writelines(strings)

    else:
        raise ValueError(f'cultivar files already exist '
                         f'for run name: "{run_name}"!')


def make_runs(run_name, yamlfile=None, cont_cvars=True):
    """
    Create run.txt files in corresponding directories for experiment.

    Parameters
    ----------
    run_name : str
        run name for batch of experiments.
    yamlfile : str
        default None - function reads init_runame.yml file in project dirct.
        a testing yamlfile path need to be passed for testing purposes.
    cont_cvars : Bool
        Default True.
        How yaml file stores simulation cvars info.
        True: stored single number representing the total number of cultivas.
        Fals: stored specific cultivars (testing purposes).

    """
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    dirct_project = dict_setup['path_project']
    dirct_runs = os.path.join(dirct_project, 'runs', run_name)

    # only execute if no run files already exist
    filelist = get_filelist(os.path.expanduser(dirct_runs))
    if len(filelist) == 0:
        # read in dict_setup to fetch site-years info
        if dict_setup['base_path'] == 'testing':
            basepath = DATA_PATH
            fpath_siteyears = os.path.join(basepath,
                                           *dict_setup['siteyears'])
            fpath_sitesummary = os.path.join(DATA_PATH,
                                             *dict_setup['site_summary'])
            dirct_init_wea = os.path.join(DATA_PATH,
                                          *dict_setup['path_wea'])

        else:
            fpath_siteyears = os.path.join(dict_setup['path_project'],
                                           *dict_setup['siteyears'])
            fpath_sitesummary = os.path.join(dict_setup['path_project'],
                                             *dict_setup['site_summary'])
            dirct_init_wea = os.path.join(dict_setup['path_project'],
                                          *dict_setup['path_wea'])

        df_sites = pd.read_csv(fpath_sitesummary)
        data = genfromtxt(fpath_siteyears,
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
            cvars_iter = np.arange(cvars[0])
        else:
            cvars_iter = cvars

        # assemble cultivars
        for var in cvars_iter:
            cultivar = 'var_' + str(var)
            cultivars.append(cultivar)

        # setup up directories
        dirct_init_stand = dict_setup['path_init_standards']

        dict_standard = {int(3): 'biology',
                         int(5): 'nitrogen',
                         int(9): 'drip',
                         int(10): 'water',
                         int(11): 'waterbound',
                         int(16): 'massbl'}
        dict_custom = {int(2): 'time',
                       int(4): 'climate',
                       int(8): 'management',
                       int(12): 'init'}
        dict_soils = {int(14): 'grid',
                      int(15): 'nod',
                      int(7): 'soil',
                      int(6): 'solute'}

        # itemize dictionary items into paths
        dict_standard_loop = dict_standard.copy()
        for key, value in dict_standard_loop.items():
            dict_standard_loop[key] = os.path.join(
                dirct_init_stand, f'{value}.txt') + '\n'

        # loop through siteyears
        for siteyear in siteyears:
            # setup site-specific soil directory
            site = siteyear.split('_')[0]
            year = siteyear.split('_')[1]
            texture = df_sites.query(
                f'site == {site}').texture.item()  # identify texture
            dirct_init_soils = os.path.join(dict_setup['path_init_soils'],
                                            texture)

            # itemize dict_soils items into paths
            dict_soils_loop = dict_soils.copy()
            for key, value in dict_soils_loop.items():
                dict_soils_loop[key] = os.path.join(
                    dirct_init_soils, f'{value}.txt') + '\n'

            # loop through cultivars
            for cultivar in cultivars:
                # set up custom directory
                dirct_init_custom = os.path.join(dirct_project,
                                                 'inits',
                                                 'customs',
                                                 run_name,
                                                 year,
                                                 cultivar)

                # itemize dict_custom items into paths
                dict_custom_loop = dict_custom.copy()
                for key, value in dict_custom_loop.items():
                    dict_custom_loop[key] = os.path.join(
                        dirct_init_custom, f'{value}.txt') + '\n'

                # set up output directory
                dirct_output = os.path.join(dirct_project,
                                            'sims',
                                            run_name,
                                            year,
                                            cultivar)
                dict_output = {int(17): 'out1_' + siteyear + '_' + cultivar,
                               int(18): 'out2_' + siteyear + '_' + cultivar,
                               int(19): 'out3',
                               int(20): 'out4',
                               int(21): 'out5',
                               int(22): 'out6',
                               int(23): 'massbl_' + siteyear + '_' + cultivar,
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
                                                  f'{cultivar}.txt') + '\n',
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
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    dirct_project = dict_setup['path_project']
    dirct_jobs = os.path.join(dirct_project, 'jobs', run_name)
    path_maizsim = dict_setup['path_maizsim']

    # only execute if no run files already exist
    filelist = get_filelist(os.path.expanduser(dirct_jobs))
    if len(filelist) == 0:
        years = dict_setup['specs']['years']  # fetch from init_runame.yml
        cvars = dict_setup['specs']['cvars']  # fetch from init_runame.yml

        # check if cultivar in yamlfile is continuous or not
        # False case mostly for testing purposes
        cultivars = list()
        if cont_cvars is True:
            cvars_iter = np.arange(cvars[0])
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
                str10 = 'for file in $FILES\n'
                str11 = 'do\n'
                str12 = f'\tcd {path_maizsim}\n'
                str13 = '\ttimeout 20m maizsim $file\n'
                str14 = 'done\n'

                strings = [str1, str2, str3, str4, str5, str6, str7, 
                           str8, str9, str10, str11, str12, str13, str14]

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
    dict_setup = read_inityaml(run_name, yamlfile=yamlfile)
    dirct_project = dict_setup['path_project']
    dirct_jobs = os.path.join(dirct_project, 'jobs')

    subjobs = 'subjobs_' + run_name + '.sh'

    if not os.path.exists(os.path.join(dirct_jobs, subjobs)):
        str1 = '#!/bin/bash\n'
        str2 = '\n'
        str3 = 'JOBS=' + os.path.join(dirct_jobs, run_name, '*') + '\n'
        str4 = '\n'
        str5 = 'for job in $JOBS\n'
        str6 = 'do\n'
        str7 = '    while [ `qstat | grep ach315 | wc -l` -ge 100 ]\n'
        str8 = '    do\n'
        str9 = '        sleep 1\n'
        str10 = '    done\n'
        str11 = '    qsub $job\n'
        str12 = 'done\n'

        strings = [str1, str2, str3, str4, str5, str6,
                   str7, str8, str9, str10, str11, str12]

        subjobs_script = open(os.path.join(dirct_jobs, subjobs), 'w')
        subjobs_script.writelines(strings)
        subjobs_script.close()

    else:
        raise ValueError(f'subjobs.sh for run_name: "{run_name}"'
                         ' already exists!')
