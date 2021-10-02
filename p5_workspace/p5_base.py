import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from neorl import ES
import string
import random

sys.path.append("..")

from fitness_help import FitnessHelper, Objective
from reactivity_model import ReactivityModel
from qpower_model import QPowerModel

def rid():
    N = 7
    return ''.join(random.choice(string.ascii_lowercase
        + string.digits) for _ in range(N))

def make_objs():
    a = ReactivityModel()
    #reactivity objective
    t_react = .03072
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
    return [tgt_react, tgt_splits, minmax_dist]

def calc_cumavg(data, N):
    """
    data: vector of FOM to plot (e.g. fitness)
    N: number of data points to group before calculating the statistics (e.g. population size)
    """

    cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
    cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
    cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
    cum_min=[np.min(data[i:i+N]) for i in range(0,len(data),N)]

    return cum_aves, cum_std, cum_max, cum_min

def plot_progress(fit_vals, n_steps):
    """
    fit_vals: NEORL predicted fitness values in numpy vector
    n_steps: population size, e.g. npop, nwolves, nwhales, etc.
    pngname: figure name
    """
    plt.figure()
    ravg, rstd, rmax, rmin=calc_cumavg(fit_vals, n_steps)
    epochs=np.array(range(1,len(ravg)+1),dtype=int)
    plt.plot(epochs, ravg,'-o', c='g', label='Average per generation')

    plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
    alpha=0.2, edgecolor='g', facecolor='g', label=r'$1-\sigma$ per generation')

    plt.plot(epochs, rmax,'s', c='k', label='Max per generation', markersize=4)
    plt.plot(epochs,rmin,'d', c='k', label='Min per generation', markersize=4)
    plt.legend()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
