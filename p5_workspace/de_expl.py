import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import DE
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

def de_expl(fevals, v = False):
    fname = Path("log/de_%s.log"%rid())
    objs = make_objs() #in order react, psplits, dist

    wts = [0.5, 0.4, 0.1]

    BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 8)}

    npop = 20
    F = 0.4
    CR = 0.3
    notes_str = "npop=%i,F=%f,CR=%f\n"%(npop, F, CR)
    de_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
    de = DE(mode="min", bounds = BOUNDS, fit = de_helper.fitness, npop=npop,
            F=F, CR=CR)
    de_x, de_y, de_hist = de.evolute(fevals//(2*npop), verbose = v)
    res = get_log(fname)
    de_helper.close()
    return de_x, de_y, de_hist, res, npop

if __name__ == "__main__":
    de_x, de_y, de_hist, res, npop = de_expl(10000, v = True)
    print("x best", np.array(de_x)*180/np.pi)
    print("y best", de_y)


    plot_progress(res["fitness"], npop)
    plot_objs(res)
    plt.show()
