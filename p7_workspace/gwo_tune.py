import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import GWO
from neorl.tune import GRIDTUNE
import random
import pandas as pd

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p7_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

#perform tune or just print current results?
tune = True
#tuning logfile
tlog = Path("log/gwo_tune.log")

#get and configure objective function
fname = Path("log/gwo_%s.log"%rid()) #logger name for objecrive function
objs = make_objs() #in order react, psplits, dist
wts = [0.5, 0.3, 0.2]
BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 9)}
notes_str = "parameter optimization"
GWO_helper = FitnessHelper(objs, wts, fname, notes = notes_str)

#define fitness function
def tune_fit(nwolves):
    gwo = GWO(mode="min", bounds = BOUNDS, fit = GWO_helper.fitness, nwolves = nwolves)
    gwo_x, gwo_y, gwo_hist = gwo.evolute(10000//nwolves)
    return gwo_y

#provide parameter grids
param_grid = {
        "nwolves"      : [5, 10, 15, 20, 25, 30, 35, 45, 50]}

gtune = GRIDTUNE(param_grid = param_grid, fit = tune_fit)
if tune:
    gridres = gtune.tune(ncores = 1, csvname=tlog)

try:
    res = pd.read_csv(tlog, index_col = 0)
    sort = res.sort_values("score")
    print(sort)
except:
    raise Exception("No tuning logfile found with name %s"%(str(tlog)))

