#--------------------------------------------------------
#Problem 8
#  Maximum differential worth, no constraints
#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, ES, PSO, PESA2
import sys
sys.path.append("..")

from reactivity_model import ReactivityModel
from qpower_model import QPowerModel

rmodel = ReactivityModel()

def fitness(x):
    x = np.asarray(x)

    gradworth = rmodel.evalg(x)
    return np.abs(gradworth).sum()


#set up bounds
BOUNDS = {"x%i"%i : ["float", -np.pi, np.pi] for i in range(1, 9)}

#optimize
#  Differential evolution
print("Starting DE...")
de = DE(mode = "max", bounds = BOUNDS, fit = fitness, npop=50,
        CR = 0.5, F = 0.7, ncores = 1)
de_x, de_y, de_hist = de.evolute(ngen = 40)



ans = {"Differential Evolution" : [de_x, de_hist]}

for key, value in ans.items():
    v = np.asarray(value[0])
    print("\n-----------------------\n", key, "\n-----------------------")
    s =["%.1f"%(a*180/np.pi) for a in v]
    print("Angles", " ".join(s))
    print("Diff worth", np.abs(rmodel.evalg(v)).sum(), "1/rad")
    print("Diff worth", np.abs(rmodel.evalg(v)).sum()*1e5/(180/np.pi), "pcm/deg")
    print("Diff worth", np.abs(rmodel.evalg(v)).sum()*1e5, "pcm/rad")


#3800 pcm/rad maximum worth
