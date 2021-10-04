import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import GWO
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

fname = Path("log/de_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

a1 = 0.98
wts = [a1, 1-a1 - 0.01, .01]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

nwolves = 10
notes_str = "nwolves=%i\n"%(nwolves)
gwo_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
gwo = GWO(mode="min", bounds = BOUNDS, fit = gwo_helper.fitness, nwolves = nwolves)
gwo_x, gwo_y, gwo_hist = gwo.evolute(1000)

res = get_log(fname)

plot_progress(res["fitness"], nwolves)
plot_objs(res)
plt.show()
