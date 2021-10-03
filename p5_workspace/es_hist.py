#--------------
# Script to generate histograms to show 
# different local minimums that occur
#--------------




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

objs = make_objs() #in order react, psplits, dist

a1 = 0.86
wts = [a1, 1-a1 - 0.01, .01]

BOUNDS = {"x%i"%i : ["float", -1.1*np.pi, 1.1*np.pi] for i in range(1, 8)}

lambda_ = 60
mu = 30
cxpb = 0.6
mutpb = 0.3
notes_str = "lambda=%i, mu=%i, cxpb=%f, mutpb=%f\n"%(lambda_, mu, 
        cxpb, mutpb)

#histname = "log/hist0p9"
histname = "log/hist0p86"

rlist = []
objectives = []
drumangles = []
I = 1000
for i in range(I):
    print(i+1, "/", I)
    fname = Path("log/es_%s.log"%rid())
    es_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
    es = ES(mode="min", bounds = BOUNDS, fit = es_helper.fitness,
            ncores=1, lambda_ = lambda_, mu = mu, cxpb = cxpb, mutpb = mutpb)
    es_x, es_y, es_hist = es.evolute(150)

    res = get_log(fname)
    bi = np.argmin(res["fitness"].values)
    rlist.append(res["react_err_obj"].values[bi]*1e5)
    objectives.append(res["fitness"].values[bi])
    drumangles.append(res.values[bi, 1:8]*180/np.pi)
    with open(histname + ".dat", "a") as f:
        f.write("%.3f,%.6f"%(rlist[-1], objectives[-1]))
        for ii in range(7):
            f.write(",%.3f"%drumangles[-1][ii])
        f.write("\n")

#plt.hist(rlist, bins= "auto", color = "#0504aa", rwidth = 0.85)
#plt.xlabel("Reactivity Error")
#plt.ylabel("Freq")
#plt.show()
#print(np.asarray(es_x)*180/np.pi)
#print(es_y)

#plot_progress(res["fitness"], lambda_)
#plot_objs(res)
#plt.show()
