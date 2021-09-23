#--------------------------------------------------------
#Problem 1
#  How can we rotate all eight drums (in unison) to achieve
#  some desired criticality?
#--------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from neorl import DE, ES

from reactivity_model import ReactivityModel

#accept target
target_reactivity = 1000 #pcm

#set up fitness function
model = ReactivityModel()

def fitness(x):
    #x is size 1 numpy array, bounded by [-pi, pi]
    thetas = np.zeros(8) + x
    return np.abs(target_reactivity*1e-5 - model.eval(thetas))

#set up bounds
BOUNDS = {"x1" : ["float", -np.pi, np.pi]}

#optimize
#  Differential evolution
de = DE(mode = "min", bounds = BOUNDS, fit = fitness, npop=50,
        CR = 0.5, F = 0.7, ncores = 1)
de_x, de_y, de_hist = de.evolute(ngen = 10)

#  Evolution strategies
es = ES(mode = "min", bounds = BOUNDS, fit = fitness, lambda_ = 40,
        mu = 30, ncores = 1)
es_x, es_y, es_hist = de.evolute(ngen = 10)

t = np.linspace(-np.pi, np.pi, 200)
p = np.zeros_like(t)

for i, tt in enumerate(t):
    p[i] = fitness(tt)

plt.plot(t, p, "k", label = "Objective Function")
plt.scatter(de_x, de_y, label = "Differential Evolution")
plt.scatter(es_x, es_y, label = "Evolution Strategies")
plt.xlabel("Drum Rotation Angle [rad.]")
plt.ylabel("Objective Function")
plt.show()
