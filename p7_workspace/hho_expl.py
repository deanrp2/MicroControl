import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import HHO
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p7_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

fname = Path("log/hho_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

wts = [0.5, 0.3, 0.2]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 9)}

nhawks = 10
notes_str = "nhawks=%i\n"%(nhawks)
hho_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
hho = HHO(mode="min", bounds = BOUNDS, fit = hho_helper.fitness,
        ncores=1, nhawks = nhawks)
hho_x, hho_y, hho_hist = hho.evolute(int(1000//(1.7*nhawks)))
print("x best", np.array(hho_x)*180/np.pi)
print("y best", hho_y)

hho_helper.close()

res = get_log(fname)

plot_progress(res["fitness"], nhawks)
plot_objs(res)
plt.show()