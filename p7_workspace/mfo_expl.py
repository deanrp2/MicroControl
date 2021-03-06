import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import MFO
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p7_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

def mfo_expl(fevals, v = False):
    fname = Path("log/mfo_%s.log"%rid())
    objs = make_objs() #in order react, psplits, dist

    wts = [0.55, 0.4, 0.05]

    BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 9)}

    nmoths = 30
    b = 1
    notes_str = "nmoths=%i\n"%(nmoths)
    mfo_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
    mfo= MFO(mode="min", bounds = BOUNDS, fit = mfo_helper.fitness, nmoths = nmoths)
    mfo_x, mfo_y, mfo_hist = mfo.evolute(fevals//nmoths, verbose = v)
    res = get_log(fname)
    return mfo_x, mfo_y, mfo_hist, res, nmoths

if __name__ == "__main__":
    mfo_x, mfo_y, mfo_hist, res, nmoths = mfo_expl(10000, v = True)

    print("x best", np.array(mfo_x)*180/np.pi)
    print("y best", mfo_y)

    plot_progress(res["fitness"], nmoths)
    plot_objs(res)
    plt.show()
