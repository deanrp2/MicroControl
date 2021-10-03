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

a1 = 0.9
wts = [a1, 1-a1 - 0.01, .01]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

config = {"pop_size" : 75,
          "num_hidden" : 2,
          "activation_mutate_rate" : 0.1,
          "survival_threshold" : 0.3}

pop = 50
CR = 0.5
F = 0.7
notes_str = str(config) + "\npop=%i, CR=%f, F=%f\n"%(pop, CR, F)
es_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
es = ES(mode="min", bounds = BOUNDS, fit = es_helper.fitness, npop =pop, CR=CR, F=F,
        ncores=1)
es_x, es_y, es_hist = es.evolute(40)

res = get_log(fname)

plot_progress(res["fitness"], pop)
plot_objs(res)
plt.show()
