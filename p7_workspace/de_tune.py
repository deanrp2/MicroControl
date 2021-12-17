import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import DE
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
tlog = Path("log/de_tune.log")

#get and configure objective function
fname = Path("log/de_%s.log"%rid()) #logger name for objecrive function
objs = make_objs() #in order react, psplits, dist
wts = [0.5, 0.3, 0.2]
BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 9)}
notes_str = "parameter optimization"
de_helper = FitnessHelper(objs, wts, fname, notes = notes_str)

#define fitness function
def tune_fit(npop, F, CR):
    de = DE(mode="min", bounds = BOUNDS, fit = de_helper.fitness, npop=npop,
            F=F, CR=CR)
    de_x, de_y, de_hist = de.evolute(10000//(2*npop))
    return de_y

#provide parameter grids
param_grid = {
        "npop" : [10, 20, 30, 35, 40, 45, 50],
        "F"    : [.4, .5, .6, .7, .8, .9],
        "CR"   : [.1, .2, .3]}

gtune = GRIDTUNE(param_grid = param_grid, fit = tune_fit)
if tune:
    gridres = gtune.tune(ncores = 1, csvname=tlog)

try:
    res = pd.read_csv(tlog, index_col = 0)
    sort = res.sort_values("score")
    print(sort)
except:
    raise Exception("No tuning logfile found with name %s"%(str(tlog)))

