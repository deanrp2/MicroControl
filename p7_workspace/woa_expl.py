import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import WOA
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p7_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

fname = Path("log/woa_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

wts = [0.5, 0.3, 0.2]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 9)}

nwhales = 10
notes_str = "nwhales=%i\n"%(nwhales)
woa_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
woa = WOA(mode="min", bounds = BOUNDS, fit = woa_helper.fitness, nwhales = nwhales)
woa_x, woa_y, woa_hist = woa.evolute(100)

res = get_log(fname)

plot_progress(res["fitness"], nwhales)
plot_objs(res)
plt.show()
