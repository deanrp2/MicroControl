import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import GWO
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p7_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

def gwo_expl(fevals, v = False):
    fname = Path("log/gwo_%s.log"%rid())
    objs = make_objs() #in order react, psplits, dist

    wts = [0.55, 0.4, 0.05]

    BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 9)}

    nwolves = 30
    notes_str = "nwolves=%i\n"%(nwolves)
    gwo_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
    gwo = GWO(mode="min", bounds = BOUNDS, fit = gwo_helper.fitness, nwolves = nwolves)
    gwo_x, gwo_y, gwo_hist = gwo.evolute(fevals//nwolves, verbose = v)
    res = get_log(fname)
    gwo_helper.close()
    return gwo_x, gwo_y, gwo_hist, res, nwolves

if __name__ == "__main__":
    gwo_x, gwo_y, gwo_hist, res, nwolves = gwo_expl(10000, v = True)

    print("x best", np.array(gwo_x)*180/np.pi)
    print("y best", gwo_y)

    plot_progress(res["fitness"], nwolves)
    plot_objs(res)
    plt.show()
