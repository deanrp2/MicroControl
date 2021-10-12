import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import MFO
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

fname = Path("log/mfo_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

wts = [0.6, 0.2, 0.1]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

nmoths= 35
b = 1
notes_str = "nmoths=%i,b=%i\n"%(nmoths, b)
mfo_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
mfo = MFO(mode="min", bounds = BOUNDS, fit = mfo_helper.fitness, nmoths = nmoths, b = b)
mfo_x, mfo_y, mfo_hist = mfo.evolute(10000//nmoths)

res = get_log(fname)

plot_progress(res["fitness"], nmoths)
plot_objs(res)
plt.show()
