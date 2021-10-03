import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import ES
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective, get_log

from p5_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

fname = Path("log/es_%s.log"%rid())
objs = make_objs() #in order react, psplits, dist

a1 = 0.92
wts = [a1, 1-a1 - 0.01, .01]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

lambda_ = 60
mu = 30
cxpb = 0.6
mutpb = 0.3
notes_str = "lambda=%i, mu=%i, cxpb=%f, mutpb=%f\n"%(lambda_, mu, 
        cxpb, mutpb)
es_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
es = ES(mode="min", bounds = BOUNDS, fit = es_helper.fitness,
        ncores=1, lambda_ = lambda_, mu = mu, cxpb = cxpb, mutpb = mutpb)
es_x, es_y, es_hist = es.evolute(150)

res = get_log(fname)

plot_progress(res["fitness"], lambda_)
plot_objs(res)
plt.show()