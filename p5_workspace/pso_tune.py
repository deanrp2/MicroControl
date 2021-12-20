import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import PSO
from neorl.tune import GRIDTUNE
import random
import pandas as pd

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

#perform tune or just print current results?
tune =False
#tuning logfile
tlog = Path("log/tune_pso.dat")

#get and configure objective function
fname = Path("log/pso_%s.log"%rid()) #logger name for objecrive function
objs = make_objs() #in order react, psplits, dist
wts = [0.5, 0.4, 0.1]
BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 8)}
notes_str = "parameter optimization"
pso_helper = FitnessHelper(objs, wts, fname, notes = notes_str)

#define fitness function
def tune_fit(npar, c1, c2, speed_mech):
    pso = PSO(mode="min", bounds = BOUNDS, fit = pso_helper.fitness, npar=npar, c1 = c1,
            c2 = c2, speed_mech = speed_mech)
    pso_x, pso_y, pso_hist = pso.evolute(10000//(1+npar))
    return pso_y

#provide parameter grids
param_grid = {
        "npar"    : [30, 40, 50],
        "c1"    : [2.05, 2.10, 2.15],
        "c2"    : [2.05, 2.10, 2.15],
        "speed_mech" : ["constric", "timew", "globw"]}

gtune = GRIDTUNE(param_grid = param_grid, fit = tune_fit)
if tune:
    gridres = gtune.tune(ncores = 1, csvname=tlog)

try:
    res = pd.read_csv(tlog, index_col = 0)
    sort = res.sort_values("score")
    print(sort)
except:
    raise Exception("No tuning logfile found with name %s"%(str(tlog)))

