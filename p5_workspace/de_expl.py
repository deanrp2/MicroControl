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

fname = Path("log/de_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

a1 = 0.92
wts = [a1, 1-a1 - 0.01, .01]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

npop = 50
F = 0.5
CR = 0.3
notes_str = "npop=%i,F=%f,CR=%f\n"%(npop, F, CR)
de_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
de = DE(mode="min", bounds = BOUNDS, fit = de_helper.fitness, npop=npop,
        F=F, CR=CR)
de_x, de_y, de_hist = de.evolute(100)

res = get_log(fname)

plot_progress(res["fitness"], npop)
plot_objs(res)
plt.show()
