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
        thetas = np.asarray(x)
        react = a.eval(thetas)
        return np.abs(react - t_react)
    minn = 0
    maxx = max([rtgt(np.zeros(8)), rtgt(np.zeros(8)+np.pi)])
    tgt_react = Objective("react_err", "min", 8, rtgt, minn,  maxx)

    #power split objective
    t_splits = np.array([0.255, 0.248333, 0.248333, 0.248333])

    b = QPowerModel()

    xmax = np.zeros(8)
    xmax[[0, 1]] += np.pi
    off_splits = b.eval(xmax)

    max_off = np.abs(off_splits - t_splits).sum()
    def qpower(x):
        thetas = np.asarray(x)
        predicted = b.eval(thetas)
        return np.abs(predicted - t_splits).sum()
    tgt_splits = Objective("psplit_err", "min", 7, qpower, 0, max_off)

    #differential worth objective
    def diffworth(x):
        thetas = np.asarray(x)
        return np.abs(a.evalg(thetas)).sum()
    diff_worth = Objective("diff_worth", "max", 8, diffworth, 0, 3800)

    return [tgt_react, tgt_splits, diff_worth]

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

def plot_progress(fit_vals, n_steps, theme = "g", ax = None,
        legend = True, m = 4):
    """
    fit_vals: NEORL predicted fitness values in numpy vector
    n_steps: population size, e.g. npop, nwolves, nwhales, etc.
    pngname: figure name
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (6, 6))

        ravg, rstd, rmax, rmin=calc_cumavg(fit_vals, n_steps)
    epochs=np.array(range(1,len(ravg)+1),dtype=int)
    l1 = ax.plot(epochs, ravg,'-o', c=theme, label='Average per generation', markersize = m,
            linewidth = .8)

    l2 = ax.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
        alpha=0.2, edgecolor=theme, facecolor=theme, label=r'$1-\sigma$ per generation')

    l3 = ax.plot(epochs, rmax,'s', c='gray', label='Max per generation', markersize=m)
    l4 = ax.plot(epochs,rmin,'d', c='k', label='Min per generation', markersize=m)
    if legend is True:
        ax.legend()

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    return l1[0], l2, l3[0], l4[0]


#TODO: update this function
def plot_objs(res, ax =  None, c = "k"):
    def cumargmin(a):
        a = -a
        m = np.maximum.accumulate(a)
        x = np.arange(a.shape[0])
        x[1:] *= m[:-1] < m[1:]
        np.maximum.accumulate(x, axis=0, out=x)
        return x
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize = (8, 6))
        ax = ax.flatten()
    titles = ["react_err_obj", "psplit_err_obj", "diff_worth_obj",
            "fitness"]
    m = cumargmin(res["fitness"].values)
    for i in range(4):
        ax[i].set_title(titles[i])
        if titles[i] == "react_err_obj":
            v = res[titles[i]]*1e5
        elif titles[i] == "diff_worth_obj":
            v = res[titles[i]]*np.pi/180*1e5
        else:
            v = res[titles[i]]
        ax[i].plot(res.index, v, c, linewidth = 1, alpha = .4)
        ax[i].plot(res.index, v.iloc[m], c,
            linewidth = 3)
    plt.tight_layout()
    return ax













