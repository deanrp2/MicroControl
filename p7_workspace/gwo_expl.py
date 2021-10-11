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

fname = Path("log/gwo_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

wts = [0.5, 0.3, 0.2]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 9)}

nwolves = 10
notes_str = "nwolves=%i\n"%(nwolves)
gwo_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
gwo = GWO(mode="min", bounds = BOUNDS, fit = gwo_helper.fitness, nwolves = nwolves)
gwo_x, gwo_y, gwo_hist = gwo.evolute(50)

res = get_log(fname)

plot_progress(res["fitness"], nwolves)
plot_objs(res)
plt.show()