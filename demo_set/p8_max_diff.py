#--------------------------------------------------------
#Problem 8
#  Maximum differential worth, no constraints
#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, ES, PSO, PESA2

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
de = DE(mode = "max", bounds = BOUNDS, fit = fitness, npop=50,
        CR = 0.5, F = 0.7, ncores = 1)
de_x, de_y, de_hist = de.evolute(ngen = 100)

#  Evolution strategies
es = ES(mode = "max", bounds = BOUNDS, fit = fitness, lambda_ = 40,
        mu = 30, ncores = 1)
es_x, es_y, es_hist = de.evolute(ngen = 100)

# Modern pesa
psa = PESA2(mode = "max", bounds = BOUNDS, fit = fitness)
psa_x, psa_y, psa_hist = psa.evolute(ngen = 100)


ans = {"Differential Evolution" : [de_x, de_hist],
        "Evolution Strategies" : [es_x, es_hist],
        "PESA2"                : [psa_x, psa_hist]}

for key, value in ans.items():
    plt.plot(value[1], label = key)
    v = np.asarray(value[0])
    print("\n-----------------------\n", key, "\n-----------------------")
    s =["%.1f"%(a*180/np.pi) for a in v]
    print("Angles", " ".join(s))
    print("Diff worth", np.abs(rmodel.evalg(v)).sum()*1e5/(180/np.pi), "pcm/deg")
    print("Diff worth", np.abs(rmodel.evalg(v)).sum()*1e5, "pcm/rad")

plt.legend()
plt.xlabel("Generations")
plt.ylabel("Obj")
plt.show()

#3800 pcm/rad maximum worth
