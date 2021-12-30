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

def es_expl(fevals, seed = None, v = False):
    fname = Path("log/es_%s.log"%rid())
    objs = make_objs() #in order react, psplits, dist

    wts = [0.55, 0.4, 0.05]

    BOUNDS = {"x%i"%i : ["float", -1.*np.pi, 1.*np.pi] for i in range(1, 9)}

    lambda_ = 50
    mu = 30
    cxpb = 0.7
    mutpb = 0.3
    notes_str = "lambda=%i, mu=%i, cxpb=%f, mutpb=%f\n"%(lambda_, mu, 
        cxpb, mutpb)
    es_helper = FitnessHelper(objs, wts, fname, notes = notes_str)
    es = ES(mode="min", bounds = BOUNDS, fit = es_helper.fitness, seed = seed,
        ncores=1, lambda_ = lambda_, mu = mu, cxpb = cxpb, mutpb = mutpb)
    es_x, es_y, es_hist = es.evolute(fevals//lambda_ - 1, verbose = v)
    es_helper.close()
    res = get_log(fname)
    return es_x, es_y, es_hist, res, lambda_

if __name__ == "__main__":
    es_x, es_y, es_hist, res, lambda_ = es_expl(10000, v = True)
    print("x best", np.array(es_x)*180/np.pi)
    print("y best", es_y)

    plot_progress(res["fitness"], lambda_)
    plot_objs(res)
    plt.show()
