import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import PESA2
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

fname = Path("log/pesa_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

wts = [0.6, 0.2, 0.1]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

npop = 50
F = 0.5
CR = 0.3
nwolves = 5
nwhales = 10
notes_str = "npop=%i,F=%f,CR=%f,nwolves=%i,nwhales=%i\n"%(npop, F, CR,nwolves,nwhales)
pesa_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
pesa = PESA2(mode="min", bounds = BOUNDS, fit = pesa_helper.fitness, npop=npop,
        F=F, CR=CR,nwolves=nwolves,nwhales=nwhales)
pesa_x, pesa_y, pesa_hist = pesa.evolute(30)

res = get_log(fname)

plot_progress(res["fitness"], npop)
plot_objs(res)
plt.show()
