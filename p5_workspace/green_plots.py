import numpy as np
import matplotlib.pyplot as plt

from de_expl import de_expl
from es_expl import es_expl
from gwo_expl import gwo_expl
from hho_expl import hho_expl
from mfo_expl import mfo_expl
from pso_expl import pso_expl
from woa_expl import woa_expl

from p5_base import plot_progress, plot_objs

fevals = 10000

#de_x, de_y, de_hist, de_res, de_npop = de_expl(fevals)
#es_x, es_y, es_hist, es_res, es_lambda_ = es_expl(fevals)
#gwo_x, gwo_y, gwo_hist, gwo_res, gwo_nwolves = gwo_expl(fevals)
#hho_x, hho_y, hho_hist, hho_res, hho_nhawks = hho_expl(fevals)
#mfo_x, mfo_y, mfo_hist, mfo_res, mfo_nmoths = mfo_expl(fevals)
#pso_x, pso_y, pso_hist, pso_res, pso_npar = pso_expl(fevals)
#woa_x, woa_y, woa_hist, woa_res, woa_nhawks = woa_expl(fevals)

fs = [de_expl, es_expl, gwo_expl, hho_expl, mfo_expl, pso_expl, woa_expl]
all_names = ["DE", "ES", "GWO", "HHO", "MFO", "PSO", "WOA"]

places = [0,1,2,3,4,5] #which methods to include

es = [fs[i](fevals) for i in places]
names = [all_names[i] for i in places]



fig, ax = plt.subplots(3, 2, sharey = True, figsize = (6, 7))
ax = ax.flatten()

for i in range(6):
    handles = plot_progress(es[i][3]["fitness"], es[i][4], ax = ax[i], legend = False,
            m = 1.3)
    if i < 4:
        ax[i].set_xlabel("")
    if i % 2 == 1:
        ax[i].set_ylabel("")
    ax[i].set_title(all_names[i])

lbls = ["gen. ave.", r"gen. 1-$\sigma$", "gen. max", "gen. min"]
fig.legend(handles, lbls, loc="upper center", ncol = 4)#, bbox_to_anchor=(2,0), loc = "lower right")
plt.subplots_adjust(top = 0.9, bottom = .09, left=0.1, right=.97, wspace=0.07, hspace=0.35)
#plt.tight_layout()


plt.show()
