import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from de_expl import de_expl
from es_expl import es_expl
from gwo_expl import gwo_expl
from hho_expl import hho_expl
from mfo_expl import mfo_expl
from pso_expl import pso_expl
from woa_expl import woa_expl

from p7_base import plot_progress, plot_objs

fevals = 7000
runs = 50

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

es = []
for ri in range(runs):
    print("Currently on run", ri, "/", runs)
    es.append([fs[i](fevals) for i in places])

print("Calcs done!")
#names = [all_names[i] for i in places]

table = open("results_table.txt", "w")
table.write("fevals: %i\n"%fevals)
table.write("runs: %i\n"%runs)

for i in range(len(places)):
    outss = [k[i] for k in es]
    xs = np.zeros((runs, 8))
    ys = np.zeros(runs)
    fc = np.zeros(runs)
    fp = np.zeros(runs)
    fw = np.zeros(runs)

    for j in range(runs):
        xs[j,:], ys[j], _, res, _ = outss[j]
        ind_best = res["fitness"].values.argmin()
        fc[j] = res["react_err_obj"].values[ind_best]
        fp[j] = res["psplit_err_obj"].values[ind_best]
        fw[j] = res["diff_worth_obj"].values[ind_best]
    xs = xs*180/np.pi
    fc *= 1e5
    fw = fw*np.pi/180*1e5

    table.write(all_names[i] + "\n")
    table.write("xs mean")
    [table.write(" %E"%a) for a in xs.mean(0)]
    table.write("\nxs std")
    [table.write(" %E"%a) for a in xs.std(0)]
    table.write("\ny mean %E\n"%ys.mean())
    table.write("y std %E\n"%ys.std())
    table.write("fc mean %E\n"%fc.mean())
    table.write("fc std %E\n"%fc.std())
    table.write("fp mean %E\n"%fp.mean())
    table.write("fp std %E\n"%fp.std())
    table.write("fw mean %E\n"%fw.mean())
    table.write("fw std %E\n"%fw.std())
table.close()
