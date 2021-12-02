import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import MFO
from neorl.tune import GRIDTUNE
import random
import pandas as pd

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

#perform tune or just print current results?
tune = True
#tuning logfile
tlog = Path("log/tune_mfo.dat")

#get and configure objective function
fname = Path("log/mfo_%s.log"%rid()) #logger name for objecrive function
objs = make_objs() #in order react, psplits, dist
wts = [0.5, 0.4, 0.1]
BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}
notes_str = "parameter optimization"
MFO_helper = FitnessHelper(objs, wts, fname, notes = notes_str)

#define fitness function
def tune_fit(nmoths):
    mfo = MFO(mode="min", bounds = BOUNDS, fit = MFO_helper.fitness, nmoths = nmoths)
    mfo_x, mfo_y, mfo_hist = mfo.evolute(10000//nmoths)
    return mfo_y

#provide parameter grids
param_grid = {
        "nmoths"      : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]}

gtune = GRIDTUNE(param_grid = param_grid, fit = tune_fit)
if tune:
    gridres = gtune.tune(ncores = 1, csvname=tlog)

try:
    res = pd.read_csv(tlog, index_col = 0)
    sort = res.sort_values("score")
    print(sort)
except:
    raise Exception("No tuning logfile found with name %s"%(str(tlog)))

