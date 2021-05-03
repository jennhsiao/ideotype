"""Read in maizsim simulation outputs."""

from itertools import compress
import os

import pandas as pd

from ideotype.utils import get_filelist


def read_sims(path):
    """
    Read in all maizsim files under a given file path.

    1. Resets column names for model output.
    2. Fetches year/site/cvar info from file name.
    3. Read in last line of model output.
    4. Document files with abnormal line output length without rearing them.
    5. Compile year/site/cvar info & last line of model output.
    6. Issues compiled into pd.DataFrame as well.

    Parameters
    ----------
    path : str
        Root path to access sims.

    Returns
    -------
    df_sims : pd.DataFrame
    df_issues : pd.DataFrame

    """
    fpaths = get_filelist(path)

    fpaths_select = [
        (fpath.split('/')[-1].split('_')[0] == 'out1') and
        (fpath.split('/')[-1].split('.')[-1] == 'txt') for fpath in fpaths]
    fpath_sims = list(compress(fpaths, fpaths_select))

    cols = ['year', 'cvar', 'site', 'date', 'jday', 'time',
            'leaves', 'mature_lvs', 'drop_lvs', 'LA', 'LA_dead', 'LAI',
            'RH', 'leaf_WP', 'PFD', 'Solrad',
            'temp_soil', 'temp_air', 'temp_can',
            'ET_dmd', 'ET_suply', 'Pn', 'Pg', 'resp', 'av_gs',
            'LAI_sunlit', 'LAI_shaded',
            'PFD_sunlit', 'PFD_shaded',
            'An_sunlit', 'An_shaded',
            'Ag_sunlit', 'Ag_shaded',
            'gs_sunlit', 'gs_shaded',
            'VPD', 'N', 'N_dmd', 'N_upt', 'N_leaf', 'PCRL',
            'dm_total', 'dm_shoot', 'dm_ear', 'dm_totleaf',
            'dm_dropleaf', 'df_stem', 'df_root',
            'roil_rt', 'mx_rootdept',
            'available_water', 'soluble_c', 'note']

    data_all = []
    issues = []

    for fpath_sim in fpath_sims:
        # extrating basic file info
        year = int(fpath_sim.split('/')[-3])
        site = fpath_sim.split('/')[-1].split('_')[1]
        cvar = int(fpath_sim.split('/')[-1].split('_')[-1].split('.')[0])

        # reading in file and setting up structure
        with open(fpath_sim, 'r') as f:
            f.seek(0, os.SEEK_END)  # move pointer to end of file
            # * f.seek(offset, whence)
            # * Position computed from adding offset to a reference point,
            # * the reference point is selected by the whence argument.
            # * os.SEEK_SET (=0)
            # * os.SEEK_CUR (=1)
            # * os.SEEK_END (=2)

            try:
                # find current position (now at the end of file)
                # and count back a few positions and read forward from there
                f.seek(f.tell() - 3000, os.SEEK_SET)
                # * f.tell() returns an integer giving the file object’s
                # * current position in the file represented as number of bytes
                # * from the beginning of the file when in binary mode
                # * and an opaque number when in text mode.

                for line in f:
                    f_content = f.readlines()

                if len(f_content[-1]) == 523:  # normal character length
                    sim_output = list(f_content[-1].split(','))
                    data = [i.strip() for i in sim_output]
                    data.insert(0, year)
                    data.insert(1, cvar)
                    data.insert(2, site)
                    data_all.append(data)

                else:
                    issues.append(fpath_sim)

            except:
                print(fpath_sim)

    df_sims = pd.DataFrame(data_all, columns=cols)
    df_sims.dm_total = df_sims.dm_total.astype(float)
    df_sims.dm_ear = df_sims.dm_ear.astype(float)
    df_issues = pd.Series(issues, dtype='str')

    return df_sims, df_issues
