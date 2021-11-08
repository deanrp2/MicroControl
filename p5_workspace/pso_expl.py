import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import PSO
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

fname = Path("log/pso_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

wts = [0.6, 0.2, 0.1]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

npar = 50
c1 = 2.05
c2 = 2.15
speed_mech = "constric"
notes_str = "npar=%i, c1=%f, c2=%f, speed_mech=%s\n"%(npar, c1, c2, speed_mech)
pso_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
pso = PSO(mode="min", bounds = BOUNDS, fit = pso_helper.fitness,
        ncores=1, npar = npar, c1 = c1, c2 = c2, speed_mech=speed_mech)
pso_x, pso_y, pso_hist = pso.evolute(10000//(1+npar))
pso_helper.close()
print("x best", np.array(pso_x)*180/np.pi)
print("y best", pso_y)


res = get_log(fname)

plot_progress(res["fitness"], npar)
plot_objs(res)
plt.show()
