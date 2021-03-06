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

from p7_base import rid, make_objs, calc_cumavg, plot_progress, \
        plot_objs

objs = make_objs() #in order react, psplits, diff_worth

wts = [.55, .4, .05]

BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 9)}

lambda_ = 60
mu = 30
cxpb = 0.6
mutpb = 0.3
notes_str = "lambda=%i, mu=%i, cxpb=%f, mutpb=%f\n"%(lambda_, mu, 
        cxpb, mutpb)

histname = "log/hist_p55p4p05_2"

rlist = []
qsplit = []
diffworth = []

objectives = []
drumangles = []
I = 2000
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
    qsplit.append(res["psplit_err_obj"].values[bi])
    diffworth.append(res["diff_worth_obj"].values[bi]*np.pi/180*1e5)
    objectives.append(res["fitness"].values[bi])
    drumangles.append(res.values[bi, 1:8]*180/np.pi)
    with open(histname + ".dat", "a") as f:
        f.write("%.3f,%.5f,%.3f,%.6f"%(rlist[-1], qsplit[-1], diffworth[-1],
            objectives[-1]))
        for ii in range(7):
            f.write(",%.3f"%drumangles[-1][ii])
        f.write("\n")
    es_helper.close()

