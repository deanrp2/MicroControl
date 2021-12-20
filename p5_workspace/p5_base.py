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
    t_react = .03309
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

def plot_objs(res, ax =  None, c = "k"):
    def cumargmin(a):
        a = -a
        m = np.maximum.accumulate(a)
        x = np.arange(a.shape[0])
        x[1:] *= m[:-1] < m[1:]
        np.maximum.accumulate(x, axis=0, out=x)
        return x
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize = (8, 6), sharex = True)
        ax = ax.flatten()
    titles = ["react_err_obj", "psplit_err_obj", "tdist_obj",
            "fitness"]
    titless = ["Reactivity", "Power Split", "Travel Dist.", r"Tot. Fitness"]
    ylabs = [r"$\hat{f}_c$ [pcm]", r"$\hat{f}_p$",r"$\hat{f}_d$ [$^\circ$]",r"$F$"]
    m = cumargmin(res["fitness"].values)
    for i in range(4):
#        ax[i].set_title(titless[i])
        if titless[i] == "Reactivity":
            v = res[titles[i]]*1e5
        elif titless[i] == "Travel Dist.":
            v = res[titles[i]]*180/np.pi
        else:
            v = res[titles[i]]
        ax[i].semilogy(res.index, v, c, linewidth = 1, alpha = .4)
        ax[i].plot(res.index, v.iloc[m], c,
            linewidth = 3)
        ax[i].set_ylabel(ylabs[i])
        if i > 1:
            ax[i].set_xlabel(r"$F(\vec{x})$ Evals.")
    plt.tight_layout()
    return ax













