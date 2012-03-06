
import cProfile

import numpy as np
import pylab as pb
import csv
from var_model import VARModel
from multiprocessing import Pool
from datetime import datetime

#v = VarModel()
#v.w = np.array([0.0, 0.0])
#v.A = np.array([[0.04, 0.2], [-0.1, 0.5]])

#ts = v.sim_series(500, None)
#pb.plot(ts)
#pb.show()


def ident_model(ts):
    v2 = VARModel()
    v2.estimate(ts, [1, 30], True, 'sbc', None)
    return v2.order()


def simulate_model(t):
    v, res = t
    r = np.zeros_like(res)
    b = np.random.uniform(size = (len(r),))
    ndx = np.argsort(b)
    r[ndx] = res
#        np.random.shuffle(r)
    return v.simulate_with_residuals(r)


def read_data2():
    # read testing csv file
    data = []
    rdr = csv.reader(open("data/test2.csv", "r"))
    for line in rdr:
        l = map(lambda x: float(x), line)
        data.append(l)
    ts = np.array(data)
    return ts


def read_data3():
    # read residuals csv file (3D noise process)
    data = []
    rdr = csv.reader(open("data/test3_residuals.csv", "r"))
    for line in rdr:
        l = map(lambda x: float(x), line)
        data.append(l)
    ts = np.array(data)
    return ts


def run_parallel_sims():

    ts = read_data2()

    print("Fitting VAR model to data")
    v = VARModel()
    v.estimate(ts[:,0], [1, 30], True, 'sbc')
    res = v.compute_residuals(ts[:, 0])
    
#    cProfile.run('simulate_model((v, res))')
    
    print("Running simulations")
    t1 = datetime.now()
    
    # simulate 10000 time series (one surrogate)
    p = Pool(4)
#    sim_ts_all = p.map(ident_model, [ts[:,0]] * 10000)
    sim_ts_all = p.map(simulate_model, [(v, res)] * 100)
    
    delta = datetime.now() - t1
    
    print("DONE after %s" % (str(delta)))


def test_simulation_residuals():
    
    res = read_data3()
    print res[:10, :]
    
    v = VARModel()
    v.w = np.array([1.0, 1.4, -2.0])
    
    v.A = np.array([ [0.3, 0.0, 0.2, 0.2, 0.3, 0.05],
                     [0.1, 0.6, 0.0, 0.0, 0.1, 0.10],
                     [0.0, 0.0, 0.0, 0.4, 0.0, 0.00] ])
    
    v.U = np.array([ [1.0, 0.1, -0.2],
                     [0.0, 1.0, -0.3],
                     [0.0, 0.0,  1.0] ])
    
    ts = v.simulate_with_residuals(res[:10, :], ndisc = 0)
    print()
    print()
    print ts[:10, :]


if __name__ == '__main__':

    test_simulation_residuals()