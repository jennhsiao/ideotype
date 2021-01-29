"""Testing DB insert values."""
from numpy import genfromtxt

from ideotype import WeaData, Sims, Params

# point to test files
fpath_wea = ('/home/disk/eos8/ach315/ideotype/ideotype/data'
             '/test_data/wea/725300_1964.txt')
fpath_sim = ('/home/disk/eos8/ach315/ideotype/ideotype/data'
             '/test_data/sims/test/1964/var_6/out1_725300_1964_var_6.txt')
# TODO: update
fpath_param = ('/home/disk/eos8/ach315/ideotype/ideotype/data'
               '/test_data/params/param_test.csv')


def test_insert_weadata(ideotype_session):
    """Test if query output matches weather data."""
    query = ideotype_session.query(WeaData).filter(
        WeaData.site == '725300').filter(
            WeaData.year == 1964).order_by(
                WeaData.date, WeaData.time).limit(10)
    results = query.all()

    data = genfromtxt(fpath_wea,
                      delimiter='\t',
                      skip_header=1,
                      dtype=(int, 'U12', int, float,
                             float, float, float, int),
                      max_rows=10)

    for index, row in enumerate(data):
        obj_wea = results[index]
        assert obj_wea.site == '725300'
        assert obj_wea.jday == row[0]
        assert obj_wea.date.year == int(row[1].split('/')[-1].split("'")[0])
        assert obj_wea.time == row[2]


def test_insert_sims(ideotype_session):
    """Test if query output match sims."""
    query = ideotype_session.query(Sims).filter(
        Sims.site == '725300').filter(
            Sims.year == 1964).filter(
                Sims.cvar == 6).limit(10)
    results = query.all()
    assert len(results) == 10

    data = genfromtxt(fpath_sim,
                      delimiter=',',
                      skip_header=1,
                      dtype=tuple(['U10'] + [float]*48 + ['<U50']),
                      max_rows=10)

    for index, row in enumerate(data):
        obj_sim = results[index]
        assert obj_sim.site == '725300'
        assert obj_sim.year == int(row[0].split('/')[-1])
        assert obj_sim.DM_ear == row[40]
        assert obj_sim.pheno == row[-1].strip()


def test_insert_params(ideotype_session):
    """Test if query output match params."""
    query = ideotype_session.query(Params).filter(
        Params.cvar == 0).filter(
            Params.param == 'stayGreen')
    results = query.all()

    data = genfromtxt(fpath_param,
                      delimiter=',',
                      skip_header=1,
                      max_rows=10)

#    for index, row in enumerate(data):
#        obj_param = results[index]
#        assert obj_param.value == row[1]
