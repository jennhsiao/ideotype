"""Testing DB insert values."""
import os
from numpy import genfromtxt

from ideotype import WeaData, Sims, Params
from ideotype.data import DATA_PATH

# point to test files
fpath_wea = os.path.join(DATA_PATH, 'test_data', 'wea', '725300_1964.txt')
fpath_sim = os.path.join(DATA_PATH, 'test_data', 'sims', 'test', '1964',
                         'var_6', 'out1_725300_1964_var_6.txt')
fpath_param = os.path.join(DATA_PATH, 'test_data', 'params', 'param_test.csv')


def test_insert_params(ideotype_session):
    """Test if query output match params."""
    query = ideotype_session.query(Params).filter(
        Params.run_name == 'test').filter(
            Params.cvar == 6).filter(
                Params.param == 'staygreen')
    results = query.all()

    data = genfromtxt(fpath_param,
                      delimiter=',',
                      max_rows=10,
                      names=True,
                      dtype=(float))

    assert results[0].value == data[6]['staygreen']


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
                      names=True,
                      dtype=tuple(['U10'] + [float]*48 + ['<U50']),
                      max_rows=10)

    for index, row in enumerate(data):
        obj_sim = results[index]
        assert obj_sim.site == '725300'
        assert obj_sim.year == int(row['date'].split('/')[-1])
        assert obj_sim.DM_ear == row['earDM']
        assert obj_sim.pheno == row['Note'].strip()
