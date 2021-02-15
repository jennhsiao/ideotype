"""Alter table for existing DB."""
from sqlalchemy import create_engine


def alter_table(fpath_db):
    """Reorder sims DB order to take advantage of auto-index."""
    engine = create_engine('sqlite:///' + fpath_db)

    with engine.connect() as con:
        con.execute('PRAGMA foreign_keys=off')
        con.execute('BEGIN TRANSACTION')
        con.execute(
            'CREATE TABLE IF NOT EXISTS new_sims(\
                run_name varchar(20) not null,\
                year int not null,\
                cvar int not null,\
                site varchar(6) not null,\
                jday int not null,\
                time int not null,\
                date varchar(10),\
                leaves float,\
                leaves_mature float,\
                leaves_dropped float,\
                LA_perplant float,\
                LA_dead float,\
                LAI float,\
                leaf_wp float,\
                temp_soil float,\
                temp_air float,\
                temp_canopy float,\
                ET_dmd float,\
                ET_sply float,\
                Pn float,\
                Pg float,\
                resp float,\
                av_gs float,\
                LAI_sun float,\
                LAI_shade float,\
                PFD_sun float,\
                PFD_shade float,\
                An_sun float,\
                An_shade float,\
                Ag_sun float,\
                Ag_shade float,\
                gs_sun float,\
                gs_shade float,\
                VPD float,\
                Nitr float,\
                N_Dem float,\
                NUpt float,\
                LeafN float,\
                PCRL float,\
                DM_total float,\
                DM_shoot float,\
                DM_ear float,\
                DM_leaf float,\
                DM_stem float,\
                DM_root float,\
                AvailW float,\
                solubleC float,\
                pheno varchar(20),\
                PRIMARY KEY (run_name, year, cvar, site, jday, time),\
                FOREIGN KEY (cvar) REFERENCES params (cvar),\
                FOREIGN KEY (site) REFERENCES site_info (site))')
        con.execute(
            'INSERT INTO new_sims('
            'run_name,'
            'year,'
            'cvar,'
            'site,'
            'jday,'
            'time,'
            'date,'
            'leaves,'
            'leaves_mature,'
            'leaves_dropped,'
            'LA_perplant,'
            'LA_dead,'
            'LAI,'
            'leaf_wp,'
            'temp_soil,'
            'temp_air,'
            'temp_canopy,'
            'ET_dmd,'
            'ET_sply,'
            'Pn,'
            'Pg,'
            'resp,'
            'av_gs,'
            'LAI_sun,'
            'LAI_shade,'
            'PFD_sun,'
            'PFD_shade,'
            'An_sun,'
            'An_shade,'
            'Ag_sun,'
            'Ag_shade,'
            'gs_sun,'
            'gs_shade,'
            'VPD,'
            'Nitr,'
            'N_Dem,'
            'NUpt,'
            'LeafN,'
            'PCRL,'
            'DM_total,'
            'DM_shoot,'
            'DM_ear,'
            'DM_leaf,'
            'DM_stem,'
            'DM_root,'
            'AvailW,'
            'solubleC,'
            'pheno) '
            'SELECT '
            'year,'
            'cvar,'
            'site,'
            'run_name,'
            'jday,'
            'time,'
            'date,'
            'leaves,'
            'leaves_mature,'
            'leaves_dropped,'
            'LA_perplant,'
            'LA_dead,'
            'LAI,'
            'leaf_wp,'
            'temp_soil,'
            'temp_air,'
            'temp_canopy,'
            'ET_dmd,'
            'ET_sply,'
            'Pn,'
            'Pg,'
            'resp,'
            'av_gs,'
            'LAI_sun,'
            'LAI_shade,'
            'PFD_sun,'
            'PFD_shade,'
            'An_sun,'
            'An_shade,'
            'Ag_sun,'
            'Ag_shade,'
            'gs_sun,'
            'gs_shade,'
            'VPD,'
            'Nitr,'
            'N_Dem,'
            'NUpt,'
            'LeafN,'
            'PCRL,'
            'DM_total,'
            'DM_shoot,'
            'DM_ear,'
            'DM_leaf,'
            'DM_stem,'
            'DM_root,'
            'AvailW,'
            'solubleC,'
            'pheno '
            'FROM sims')
        con.execute('DROP TABLE sims')
        con.execute('ALTER TABLE new_sims RENAME TO sims')
        con.execute('CREATE INDEX id_year ON sims(year)')
        con.execute('CREATE INDEX id_cvar ON sims(cvar)')
        con.execute('CREATE INDEX id_site ON sims(site)')
        con.execute('CREATE INDEX id_pheno ON sims(pheno)')
