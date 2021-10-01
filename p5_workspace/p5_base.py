import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import ES

sys.path.append("..")

from fitness_help import FitnessHelper, Objective
from reactivity_model import ReactivityModel
from qpower_model import QPowerModel

a = ReactivityModel()
#reactivity objective
t_react = .04000
def rtgt(x):
    thetas = np.zeros(8)
    thetas[0] = x[0]
    thetas[2:] = x[1:]
    react = a.eval(thetas)
    return np.abs(react - t_react)
minn = 0
maxx = max([rtgt(np.zeros(7)), rtgt(np.zeros(7)+np.pi)])
tgt_react = Objective("react_err", "min", 7, rtgt, minn,  maxx)

#power split objective
t_splits = np.zeros(4) + 0.25

b = QPowerModel()

xmax = np.zeros(8)
xmax[[0, 1]] += np.pi
off_splits = b.eval(xmax)

max_off = np.abs(off_splits - t_splits).sum()
def qpower(x):
    x = np.asarray(x)
    thetas = np.zeros(8)
    thetas[0] = x[0]
    thetas[2:] = x[1:]
    predicted = b.eval(thetas)
    return np.abs(predicted - t_splits).sum()

tgt_splits = Objective("psplit_err", "min", 7, qpower, 0, max_off)

#min max travel distance objective
def tdist(x):
    return np.max(np.abs(x))
minmax_dist = Objective("tdist", "min", 7, tdist, 0, np.pi)

objs = [tgt_react, diff_worth, tgt_splits]
wts = [.33, .33, .34]
td = FitnessHelper(objs, wts, Path("log/test.log"))

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

config = {"pop_size" : 75,
          "num_hidden" : 2,
          "activation_mutate_rate" : 0.1,
          "survival_threshold" : 0.3}
print(config)
exit()

es = ES(mode="min", bounds = BOUNDS, fit = td.fitness)








