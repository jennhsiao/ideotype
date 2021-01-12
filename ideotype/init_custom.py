import pandas as pd
import glob as glob
import yaml  # might need to include in conda environment
# pyyaml - check this, install in environment


def find_dircts(dirct_all):
    """
    e.g. directory: '/home/disk/eos8/ach315/upscale/inits/opt_pdate/*'
    """
    directories = glob.glob(dirct_all)

    return directories


def read_siteinfo(file_siteinfo, file_siteyears):
    """
    e.g. file_siteinfo:
    '/home/disk/eos8/ach315/upscale/weadata/stations_info_9110.csv'
    """
    site_info = pd.read_csv(file_siteinfo,
                            dtype={'USAF': str},
                            usecols=[0, 1, 3, 4, 8, 9, 10])
    site_info.columns = ['site', 'class', 'station',
                         'state', 'tzone', 'lat', 'lon']
    siteyears = pd.read_csv(file_siteyears, dtype=str, index_col=0)

    return site_info, siteyears


# setting up tab
# tab = '    ' # you might not even need this,
# check if init files work with just \t


def write_init(file_init):
    """
    creates custom initial files
    (init.txt, time.txt, climate.txt, management.txt)
    for maizsim simulations.

    Parameters
    ----------
    file_init: str
        init_.yml file that includes experiment info & init info
        to generate inital files

    Returns
    -------
    init.txt
    time.txt
    climate.txt
    management.txt

    """
    # read in .yml file with init file parameter info
    with open(file_init, 'r') as pfile:
        param_dict = yaml.safe_load(pfile)

    dircts = find_dircts(param_dict["setup"]["path"])
    site_info, siteyears = read_siteinfo(param_dict["setup"]["site_info"],
                                         param_dict["setup"]["siteyears"])

    for dirct in dircts:
        init = open(dirct + '/init.txt', 'w')
        year = dirct.split('/')[-1].split('_')[-1]
        site = dirct.split('/')[-1].split('_')[-2]

        # customized parameters: location
        lat = site_info[site_info.site == site].lat.item()
        lon = site_info[site_info.site == site].lon.item()

        pdate = siteyears[(siteyears.site == site) &
                          (siteyears.year == year)].iloc[0, 3]
        pdate_month = pdate.split('-')[1]
        pdate_day = pdate.split('-')[2]

        # customized parameters: timing
        start = "'" + f'{param_dict["init"]["start_date"]}' + year + "'"
        sowing = "'" + pdate_month + '/' + pdate_day + '/' + year + "'"
        end = "'" + f'{param_dict["init"]["end_date"]}' + year + "'"

        # setting up text strings
        str1 = '*** initialization data ***\n'
        str2 = ('poprow\trowsp\tplant_density\trowang\t'
                'x_seed\ty_seed\ttab\tCEC\teomult\n')
        str3 = (f'{param_dict["init"]["poprow"]:.1f}\t'
                f'{param_dict["init"]["rowsp"]:.1f}\t'
                f'{param_dict["init"]["plant_density"]:.1f}\t'
                f'{param_dict["init"]["rowang"]:.1f}\t'
                f'{param_dict["init"]["x_seed"]:.1f}\t'
                f'{param_dict["init"]["y_seed"]:.1f}\t'
                f'{param_dict["init"]["cec"]:.2f}\t'
                f'{param_dict["init"]["eomult"]:.2f}\n')
        str4 = 'latitude\tlongitude\taltitude\n'
        str5 = (f'{lat:.2f}\t'
                f'{lon:.2f}\t'
                f'{param_dict["init"]["alt"]:.2f}\n')
        str6 = 'autoirrigate\n'
        str7 = f'{param_dict["init"]["irrigate"]}\n'
        str8 = 'begin\tsowing\tend\ttimestep (mins)\n'
        str9 = (start + '\t' + sowing + '\t' + end + '\t'
                f'{param_dict["init"]["timestep"]:.0f}\n')
        str10 = 'output soils data (g03, g04, g05, and g06 files) 1 if true\n'
        str11 = 'no soil files\toutputsoil files\n'
        if param_dict["init"]["soil"]:
            str12 = '0\t1\n'
        else:
            str12 = '1\t0\n'

        # compiling all strings
        strings = [str1, str2, str3, str4, str5, str6,
                   str7, str8, str9, str10, str11, str12]

        # writing out .txt file and clsoing file
        init.writelines(strings)
        init.close()


write_init('/home/disk/eos8/ach315/ideotype/ideotype/data/init_pdate.yml')
